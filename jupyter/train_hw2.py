#!/usr/bin/env python3
"""Train SasRec-I2I and ContextualRanker from botify logs.

python jupyter/train_hw2.py --logs data/coldstart --out botify/data
"""

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def read_logs(data_dir: Path) -> pd.DataFrame:
    paths = glob.glob(str(data_dir / "*/data.json")) or glob.glob(
        str(data_dir / "**/data.json"), recursive=True
    )
    if not paths:
        raise FileNotFoundError(f"No data.json under {data_dir}")
    frames = [pd.read_json(p, lines=True) for p in sorted(paths)]
    return pd.concat(frames, ignore_index=True)


def sessionize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().all():
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = (df["timestamp"].astype("int64") // 10**6).astype("int64")

    df = df.sort_values(["user", "timestamp"])
    df["session"] = df.groupby("user")["message"].transform(
        lambda s: (s == "last").astype(int).cumsum().shift(fill_value=0)
    )
    df["start"] = df.groupby(["user", "session"])["track"].transform("first")
    out = df[["user", "session", "track", "time", "timestamp", "start", "message"]].copy()
    out["user"] = out["user"].astype(int)
    out["track"] = out["track"].astype(int)
    out["start"] = out["start"].astype(int)
    out["time"] = out["time"].astype(float)
    return out.reset_index(drop=True)


def i2i_to_jsonl(pairs, path: Path, k: int = 50):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item_id, neighbors in pairs:
            f.write(
                json.dumps(
                    {
                        "item_id": int(item_id),
                        "recommendations": [int(x) for x in neighbors[:k]],
                    }
                )
                + "\n"
            )


class MiniSASRec(nn.Module):

    def __init__(self, n_items: int, dim: int = 64, n_heads: int = 4, max_len: int = 50, dropout: float = 0.2):
        super().__init__()
        self.max_len = max_len
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        b, l = seq.shape
        pos = torch.arange(l, device=seq.device).unsqueeze(0).expand(b, -1)
        x = self.dropout(self.item_emb(seq) + self.pos_emb(pos))
        key_padding = seq.eq(0)
        causal = torch.triu(torch.ones(l, l, device=seq.device, dtype=torch.bool), diagonal=1)
        out = self.encoder(x, mask=causal, src_key_padding_mask=key_padding)
        return out


def train_sasrec_i2i(interactions: pd.DataFrame, out_path: Path, k: int = 50):
    try:
        from rectools import Columns
        from rectools.dataset import Dataset
        from rectools.models import SASRecModel

        positives = interactions[interactions["time"] > 0.7].copy()
        positives = positives.rename(
            columns={
                "user": Columns.User,
                "track": Columns.Item,
                "timestamp": Columns.Datetime,
            }
        )
        positives[Columns.Weight] = 1.0
        counts = positives.groupby(Columns.User).size()
        keep_users = counts[counts >= 2].index
        positives = positives[positives[Columns.User].isin(keep_users)]
        if positives.empty:
            raise RuntimeError("Not enough positive interactions for SasRec")

        dataset = Dataset.construct(
            positives[[Columns.User, Columns.Item, Columns.Datetime, Columns.Weight]]
        )
        model = SASRecModel(
            session_max_len=50,
            loss="softmax",
            n_factors=64,
            n_blocks=1,
            n_heads=4,
            dropout_rate=0.2,
            lr=0.001,
            batch_size=256,
            epochs=20,
            verbose=1,
            deterministic=True,
        )
        print("Fitting rectools SasRec...")
        model.fit(dataset)
        item_ids = (
            positives.groupby(Columns.Item).size().sort_values(ascending=False).index.tolist()
        )
        item_ids = item_ids[: min(len(item_ids), 16000)]
        print(f"Generating SasRec I2I for {len(item_ids)} items...")
        i2i = model.recommend_to_items(
            target_items=item_ids,
            dataset=dataset,
            k=k,
            filter_itself=True,
            items_to_recommend=None,
        )
        pairs = []
        for target_item_id, group in i2i.groupby("target_item_id"):
            neighbors = group.sort_values("rank")["item_id"].tolist()
            pairs.append((target_item_id, neighbors))
        i2i_to_jsonl(pairs, out_path, k=k)
        print(f"Wrote {out_path} ({len(pairs)} items)")
        return
    except Exception as e:
        print(f"rectools SasRec unavailable ({e}); falling back to MiniSASRec")

    positives = interactions[interactions["time"] > 0.7].copy()
    sessions = []
    for (_, _), g in positives.sort_values("timestamp").groupby(["user", "session"]):
        seq = g["track"].astype(int).tolist()
        if len(seq) >= 2:
            sessions.append(seq)
    if len(sessions) < 100:
        raise RuntimeError("Not enough sessions for MiniSASRec")

    all_items = sorted({t for s in sessions for t in s})
    item2idx = {t: i + 1 for i, t in enumerate(all_items)}
    idx2item = {i: t for t, i in item2idx.items()}
    n_items = len(all_items)
    max_len = 50

    samples = []
    for seq in sessions:
        idxs = [item2idx[t] for t in seq][-max_len:]
        for i in range(1, len(idxs)):
            hist = idxs[:i][-max_len:]
            target = idxs[i]
            pad = [0] * (max_len - len(hist))
            samples.append((pad + hist, target))

    rng = np.random.RandomState(31337)
    rng.shuffle(samples)
    split = int(0.95 * len(samples))
    train_samples = samples[:split]

    def collate(batch):
        xs = torch.tensor([b[0] for b in batch], dtype=torch.long)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys

    loader = DataLoader(train_samples, batch_size=256, shuffle=True, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniSASRec(n_items, dim=64, n_heads=4, max_len=max_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training MiniSASRec on {device} ({len(train_samples)} samples)...")
    model.train()
    for epoch in range(8):
        total, n = 0.0, 0
        for xs, ys in loader:
            xs, ys = xs.to(device), ys.to(device)
            out = model(xs)[:, -1, :]
            logits = out @ model.item_emb.weight[1:].T
            loss = F.cross_entropy(logits, ys - 1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(ys)
            n += len(ys)
        print(f"  epoch {epoch+1}: ce={total / max(n,1):.4f}")

    emb = model.item_emb.weight.detach().cpu().numpy()[1:].astype(np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    print(f"Building SasRec I2I neighbors (k={k})...")
    pairs_out = []
    batch = 256
    for start in range(0, n_items, batch):
        end = min(start + batch, n_items)
        sims = emb[start:end] @ emb.T
        for local_i, item_idx in enumerate(range(start, end)):
            scores = sims[local_i].copy()
            scores[item_idx] = -1e9
            top = np.argpartition(-scores, kth=min(k, n_items - 1))[:k]
            top = top[np.argsort(-scores[top])]
            neighbors = [idx2item[int(j) + 1] for j in top]
            pairs_out.append((idx2item[item_idx + 1], neighbors))
    i2i_to_jsonl(pairs_out, out_path, k=k)
    print(f"Wrote {out_path} ({len(pairs_out)} items)")


class ContextualRanker(nn.Module):
    def __init__(self, n_items: int, dim: int = 64):
        super().__init__()
        self.context = nn.Embedding(n_items, dim)
        self.track = nn.Embedding(n_items, dim)
        nn.init.normal_(self.context.weight, std=0.01)
        nn.init.normal_(self.track.weight, std=0.01)

    def forward(self, ctx, trk):
        return (self.context(ctx) * self.track(trk)).sum(dim=1)


def train_contextual_i2i(interactions: pd.DataFrame, out_path: Path, k: int = 50, dim: int = 64):
    pairs = interactions.copy()
    pairs = pairs.dropna(subset=["start", "track", "time"])
    pairs = pairs[pairs["start"] != pairs["track"]]
    if len(pairs) < 1000:
        raise RuntimeError("Not enough pairs for ContextualRanker")

    all_tracks = sorted(set(pairs["start"].astype(int)) | set(pairs["track"].astype(int)))
    track2idx = {t: i for i, t in enumerate(all_tracks)}
    idx2track = {i: t for t, i in track2idx.items()}
    n_items = len(all_tracks)

    ctx = pairs["start"].map(track2idx).astype(np.int64).values
    trk = pairs["track"].map(track2idx).astype(np.int64).values
    y = pairs["time"].astype(np.float32).values

    rng = np.random.RandomState(31337)
    perm = rng.permutation(len(y))
    split = int(0.9 * len(y))
    train_idx, val_idx = perm[:split], perm[split:]

    def make_loader(idxs, shuffle):
        ds = TensorDataset(
            torch.from_numpy(ctx[idxs]),
            torch.from_numpy(trk[idxs]),
            torch.from_numpy(y[idxs]),
        )
        return DataLoader(ds, batch_size=4096, shuffle=shuffle)

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContextualRanker(n_items, dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)

    best_val = float("inf")
    best_state = None
    patience, bad = 5, 0
    print(f"Training ContextualRanker on {device} ({len(train_idx)} train pairs)...")
    for epoch in range(40):
        model.train()
        for c, t, target in train_loader:
            c, t, target = c.to(device), t.to(device), target.to(device)
            pred = model(c, t)
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        vals = []
        with torch.no_grad():
            for c, t, target in val_loader:
                c, t, target = c.to(device), t.to(device), target.to(device)
                vals.append(F.mse_loss(model(c, t), target).item())
        val_loss = float(np.mean(vals)) if vals else 0.0
        print(f"  epoch {epoch+1}: val_mse={val_loss:.5f}")
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    context_emb = model.context.weight.detach().cpu().numpy().astype(np.float32)
    track_emb = model.track.weight.detach().cpu().numpy().astype(np.float32)
    track_emb = track_emb / (np.linalg.norm(track_emb, axis=1, keepdims=True) + 1e-8)
    context_emb = context_emb / (np.linalg.norm(context_emb, axis=1, keepdims=True) + 1e-8)

    print(f"Building I2I neighbors (k={k})...")
    pairs_out = []
    batch = 256
    for start in range(0, n_items, batch):
        end = min(start + batch, n_items)
        sims = context_emb[start:end] @ track_emb.T
        for local_i, item_idx in enumerate(range(start, end)):
            scores = sims[local_i]
            scores[item_idx] = -1e9
            top = np.argpartition(-scores, kth=min(k, n_items - 1))[:k]
            top = top[np.argsort(-scores[top])]
            neighbors = [idx2track[int(j)] for j in top]
            pairs_out.append((idx2track[item_idx], neighbors))

    i2i_to_jsonl(pairs_out, out_path, k=k)
    print(f"Wrote {out_path} ({len(pairs_out)} items)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", required=True)
    parser.add_argument("--out", default="botify/data")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--skip-sasrec", action="store_true")
    parser.add_argument("--skip-contextual", action="store_true")
    args = parser.parse_args()

    random.seed(31337)
    np.random.seed(31337)
    torch.manual_seed(31337)

    logs = Path(args.logs)
    out = Path(args.out)
    print(f"Reading logs from {logs}...")
    raw = read_logs(logs)
    print(f"Raw events: {len(raw)}")
    interactions = sessionize(raw)
    print(f"Sessionized rows: {len(interactions)}")

    if not args.skip_contextual:
        train_contextual_i2i(interactions, out / "contextual_i2i.jsonl", k=args.k)
    if not args.skip_sasrec:
        train_sasrec_i2i(interactions, out / "sasrec_i2i.jsonl", k=args.k)


if __name__ == "__main__":
    main()
