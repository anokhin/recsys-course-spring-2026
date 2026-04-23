import argparse
import glob
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

torch = nn = F = DataLoader = Dataset = None



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs",        required=True)
    p.add_argument("--tracks",      required=True)
    p.add_argument("--output-i2i",  required=True)
    p.add_argument("--output-meta", required=True)
    p.add_argument("--topk",        type=int, default=500)
    p.add_argument("--emb-dim",     type=int, default=64)
    p.add_argument("--hidden",      type=int, default=128)
    p.add_argument("--top-genres",  type=int, default=50)
    p.add_argument("--epochs",      type=int, default=10)
    p.add_argument("--batch-size",  type=int, default=1024)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--neg-per-pos", type=int, default=4)
    p.add_argument("--min-listen",  type=float, default=0.5)
    p.add_argument("--seed",        type=int, default=31312)
    return p.parse_args()



def ensure_torch():
    global torch, nn, F, DataLoader, Dataset
    if torch is not None:
        return
    import torch as _t
    import torch.nn as _nn
    import torch.nn.functional as _f
    from torch.utils.data import DataLoader as _dl, Dataset as _ds
    torch = _t; nn = _nn; F = _f; DataLoader = _dl; Dataset = _ds


def read_logs(log_dir: Path) -> pd.DataFrame:
    paths = glob.glob(str(log_dir / "**/data.json*"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No logs in {log_dir}")
    return pd.concat([pd.read_json(p, lines=True) for p in sorted(paths)], ignore_index=True)


def _parse_year(v):
    if not v:
        return 0
    try:
        return int(str(v)[:4])
    except Exception:
        return 0


def read_tracks(path: Path):
    meta = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tid = int(r["track"])
            meta[tid] = {
                "artist_id": int(r["artist_id"]) if r.get("artist_id") is not None else -1,
                "genres":    list(r.get("genres") or []),
                "mood":      r.get("mood") or "",
                "year":      _parse_year(r.get("year")),
                "artist_fans": float(r.get("artist_fans") or 0),
            }
    return meta



def build_vocab(track_meta, top_genres=50):
    genre_freq = Counter()
    for m in track_meta.values():
        for g in m["genres"]:
            genre_freq[g] += 1

    genre_vocab = {g: i+1 for i, (g, _) in enumerate(genre_freq.most_common(top_genres))}

    mood_vocab = {}
    for m in track_meta.values():
        md = m["mood"]
        if md and md not in mood_vocab:
            mood_vocab[md] = len(mood_vocab) + 1

    artist_vocab = {}
    for m in track_meta.values():
        aid = m["artist_id"]
        if aid >= 0 and aid not in artist_vocab:
            artist_vocab[aid] = len(artist_vocab) + 1

    return genre_vocab, mood_vocab, artist_vocab


def track_features(tid, meta, genre_vocab, mood_vocab, artist_vocab, top_genres):
    m = meta.get(tid, {})

    artist_idx = artist_vocab.get(m.get("artist_id", -1), 0)

    genre_vec = np.zeros(top_genres, dtype=np.float32)
    for g in m.get("genres", []):
        idx = genre_vocab.get(g)
        if idx is not None:
            genre_vec[idx - 1] = 1.0

    mood_idx = mood_vocab.get(m.get("mood", ""), 0)
    n_moods = len(mood_vocab)
    mood_vec = np.zeros(n_moods, dtype=np.float32)
    if mood_idx > 0:
        mood_vec[mood_idx - 1] = 1.0

    year = m.get("year", 0)
    year_f = (year - 1970) / 60.0 if year else 0.0
    fans_f = math.log1p(m.get("artist_fans", 0)) / 20.0

    return artist_idx, genre_vec, mood_vec, np.array([year_f, fans_f], dtype=np.float32)



def build_pairs(df: pd.DataFrame, min_listen: float):
    events = df[df["message"].isin(["next", "last"])].sort_values(["user", "timestamp"])
    positives = []
    hard_negs  = []

    for _, grp in events.groupby("user"):
        rows = list(grp.itertuples())
        for i in range(len(rows) - 1):
            cur, nxt = rows[i], rows[i + 1]
            anchor = int(cur.track)
            target = int(nxt.track)
            anchor_time = float(cur.time)
            target_time = float(nxt.time)

            if anchor_time >= min_listen:
                if target_time >= min_listen:
                    positives.append((anchor, target, anchor_time))
                else:
                    hard_negs.append((anchor, target))

    print(f"  positives: {len(positives)}, hard_negs: {len(hard_negs)}")
    return positives, hard_negs



class PairDataset:
    def __init__(self, positives, hard_negs, all_items, track_meta,
                 genre_vocab, mood_vocab, artist_vocab, top_genres,
                 neg_per_pos, rng):
        self.positives   = positives
        self.hard_negs   = hard_negs
        self.all_items   = list(all_items)
        self.track_meta  = track_meta
        self.genre_vocab = genre_vocab
        self.mood_vocab  = mood_vocab
        self.artist_vocab = artist_vocab
        self.top_genres  = top_genres
        self.neg_per_pos = neg_per_pos
        self.rng         = rng
        self.hard_by_anchor = defaultdict(list)
        for a, b in hard_negs:
            self.hard_by_anchor[a].append(b)

    def __len__(self):
        return len(self.positives)

    def _feat(self, tid):
        return track_features(
            tid, self.track_meta, self.genre_vocab,
            self.mood_vocab, self.artist_vocab, self.top_genres
        )

    def __getitem__(self, idx):
        anchor, target, duration = self.positives[idx]
        weight = math.log1p(duration)

        half = max(1, self.neg_per_pos // 2)

        hard_pool = self.hard_by_anchor.get(anchor, [])
        if hard_pool:
            k = min(half, len(hard_pool))
            hard = self.rng.sample(hard_pool, k)
        else:
            hard = []

        n_rand = self.neg_per_pos - len(hard)
        rand_negs = []
        while len(rand_negs) < n_rand:
            c = self.rng.choice(self.all_items)
            if c != target:
                rand_negs.append(c)

        negatives = hard + rand_negs

        return anchor, target, negatives, weight



def build_two_tower(n_artists, n_genres, n_moods, emb_dim, hidden):
    def make_encoder():
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.artist_emb = nn.Embedding(n_artists + 1, 16, padding_idx=0)
                in_dim = 16 + n_genres + n_moods + 2
                self.mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, emb_dim),
                )

            def forward(self, artist_ids, genre_vecs, mood_vecs, scalars):
                a = self.artist_emb(artist_ids)
                x = torch.cat([a, genre_vecs, mood_vecs, scalars], dim=-1)
                return F.normalize(self.mlp(x), dim=-1)
        return Encoder()

    class TwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_enc = make_encoder()
            self.item_enc  = make_encoder()

        def encode_query(self, a, g, m, s):
            return self.query_enc(a, g, m, s)

        def encode_item(self, a, g, m, s):
            return self.item_enc(a, g, m, s)

    return TwoTower()



def collate_features(ids, track_meta, genre_vocab, mood_vocab, artist_vocab, top_genres, device):
    arts, genres, moods, scalars = [], [], [], []
    for tid in ids:
        ai, gv, mv, sc = track_features(tid, track_meta, genre_vocab, mood_vocab, artist_vocab, top_genres)
        arts.append(ai)
        genres.append(gv)
        moods.append(mv)
        scalars.append(sc)
    return (
        torch.tensor(arts,    dtype=torch.long,  device=device),
        torch.tensor(np.stack(genres),  dtype=torch.float32, device=device),
        torch.tensor(np.stack(moods),   dtype=torch.float32, device=device),
        torch.tensor(np.stack(scalars), dtype=torch.float32, device=device),
    )


def train(model, dataset, epochs, batch_size, lr, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start: start + batch_size]
            anchors, targets, neg_lists, weights = [], [], [], []
            for i in batch_idx:
                a, t, negs, w = dataset[i]
                anchors.append(a)
                targets.append(t)
                neg_lists.append(negs[0] if negs else random.choice(dataset.all_items))
                weights.append(w)

            wa = collate_features(anchors,  dataset.track_meta, dataset.genre_vocab,
                                  dataset.mood_vocab, dataset.artist_vocab, dataset.top_genres, device)
            wt = collate_features(targets,  dataset.track_meta, dataset.genre_vocab,
                                  dataset.mood_vocab, dataset.artist_vocab, dataset.top_genres, device)
            wn = collate_features(neg_lists, dataset.track_meta, dataset.genre_vocab,
                                  dataset.mood_vocab, dataset.artist_vocab, dataset.top_genres, device)
            wgt = torch.tensor(weights, dtype=torch.float32, device=device)

            q_emb = model.encode_query(*wa)
            p_emb = model.encode_item(*wt)
            n_emb = model.encode_item(*wn)

            pos_score = (q_emb * p_emb).sum(dim=-1)
            neg_score = (q_emb * n_emb).sum(dim=-1)

            loss = (-F.logsigmoid(pos_score - neg_score) * wgt).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss)
            n_batches  += 1

        print(f"epoch={epoch} loss={total_loss/max(n_batches,1):.4f}")



def extract_embeddings(model, all_items, track_meta, genre_vocab, mood_vocab, artist_vocab, top_genres, device):
    model.eval()
    items = sorted(all_items)
    embs  = []
    with torch.no_grad():
        for start in range(0, len(items), 512):
            chunk = items[start: start + 512]
            wa = collate_features(chunk, track_meta, genre_vocab, mood_vocab, artist_vocab, top_genres, device)
            e = model.encode_item(*wa)
            embs.append(e.cpu().numpy())
    return items, np.vstack(embs).astype(np.float32)


def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.hnsw.efConstruction = 200
    index.add(embeddings)
    index.hnsw.efSearch = 100
    return index


def retrieve_candidates(model, anchor_items, all_items, item_idx, faiss_index,
                        track_meta, genre_vocab, mood_vocab, artist_vocab, top_genres,
                        topk, device):
    model.eval()
    results = {}
    with torch.no_grad():
        for start in range(0, len(anchor_items), 256):
            chunk = anchor_items[start: start + 256]
            wa = collate_features(chunk, track_meta, genre_vocab, mood_vocab, artist_vocab, top_genres, device)
            q_emb = model.encode_query(*wa).cpu().numpy().astype(np.float32)

            _, I = faiss_index.search(q_emb, topk + 1)
            for anchor, neighbors in zip(chunk, I):
                recs = [all_items[j] for j in neighbors if all_items[j] != anchor][:topk]
                results[anchor] = recs

    return results



def main():
    args = parse_args()
    ensure_torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    print("Loading data")
    df         = read_logs(Path(args.logs))
    track_meta = read_tracks(Path(args.tracks))
    all_items  = sorted(track_meta.keys())

    print("Building vocabulary")
    genre_vocab, mood_vocab, artist_vocab = build_vocab(track_meta, top_genres=args.top_genres)
    print(f"  artists={len(artist_vocab)} genres={args.top_genres} moods={len(mood_vocab)}")

    print("Building pairs")
    positives, hard_negs = build_pairs(df, min_listen=args.min_listen)

    rng = random.Random(args.seed)
    dataset = PairDataset(
        positives, hard_negs, all_items, track_meta,
        genre_vocab, mood_vocab, artist_vocab, args.top_genres,
        neg_per_pos=args.neg_per_pos, rng=rng,
    )
    print(f"  dataset size: {len(dataset)}")

    print("Building model")
    model = build_two_tower(
        n_artists=len(artist_vocab),
        n_genres=args.top_genres,
        n_moods=len(mood_vocab),
        emb_dim=args.emb_dim,
        hidden=args.hidden,
    )

    print("Training")
    train(model, dataset, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, device=device)

    print("Extracting item embeddings")
    items, embeddings = extract_embeddings(
        model, all_items, track_meta, genre_vocab, mood_vocab, artist_vocab,
        args.top_genres, device,
    )
    item_idx = {item: i for i, item in enumerate(items)}

    print("Building FAISS HNSW index")
    index = build_faiss_index(embeddings)

    print(f"Retrieving top-{args.topk} candidates for {len(items)} anchors")
    i2i = retrieve_candidates(
        model, items, items, item_idx, index,
        track_meta, genre_vocab, mood_vocab, artist_vocab, args.top_genres,
        topk=args.topk, device=device,
    )

    out = Path(args.output_i2i)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for anchor, recs in i2i.items():
            f.write(json.dumps({"item_id": int(anchor), "recommendations": [int(r) for r in recs]}) + "\n")
    print(f"wrote {len(i2i)} anchors -> {out}")

    meta_out = Path(args.output_meta)
    with meta_out.open("wb") as f:
        pickle.dump({
            "genre_vocab":   genre_vocab,
            "mood_vocab":    mood_vocab,
            "artist_vocab":  artist_vocab,
            "top_genres":    args.top_genres,
            "items":         items,
            "embeddings":    embeddings,
            "item_idx":      item_idx,
        }, f)
    print(f"wrote meta -> {meta_out}")


if __name__ == "__main__":
    main()
