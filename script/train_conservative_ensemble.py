import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def read_catalog(path):
    tracks = []
    artists = {}
    with open(path) as f:
        for line in f:
            x = json.loads(line)
            track = int(x["track"])
            tracks.append(track)
            artists[track] = str(x.get("artist", ""))
    return tracks, artists


def read_recs(path):
    rows = []
    with open(path) as f:
        for line in f:
            x = json.loads(line)
            key = "item_id" if "item_id" in x else "track"
            rec_key = "recommendations" if "recommendations" in x else "tracks"
            rows.append((int(x[key]), [int(y) for y in x[rec_key]]))
    return rows


def read_logs(paths):
    files = []
    for root in paths:
        files.extend(Path(root).glob("**/data.json"))
    frames = []
    for file in sorted(set(files)):
        try:
            frames.append(pd.read_json(file, lines=True))
        except ValueError:
            pass
    if not frames:
        raise RuntimeError("no data.json files found")
    return pd.concat(frames, ignore_index=True)


def build_sessions(df):
    df = df.dropna(subset=["user", "track", "time", "timestamp", "message"]).copy()
    df["user"] = df["user"].astype(int)
    df["track"] = df["track"].astype(int)
    df["time"] = df["time"].astype(float)
    sessions = []
    for _, g in df.sort_values("timestamp").groupby("user"):
        cur = []
        for row in g.itertuples(index=False):
            track = int(row.track)
            time = float(row.time)
            if time >= 0:
                cur.append((track, min(time, 1.0)))
            if row.message == "last":
                if len(cur) >= 2:
                    sessions.append(cur)
                cur = []
    return sessions


def quality(df, prior):
    cnt = defaultdict(float)
    sm = defaultdict(float)
    for row in df.itertuples(index=False):
        try:
            track = int(row.track)
            time = float(row.time)
        except Exception:
            continue
        if time >= 0:
            cnt[track] += 1.0
            sm[track] += min(time, 1.0)
    mean = sum(sm.values()) / max(sum(cnt.values()), 1.0)
    q = {}
    for track in cnt:
        q[track] = (sm[track] + prior * mean) / (cnt[track] + prior)
    return q, mean


def transition_graph(sessions, window):
    graph = defaultdict(lambda: defaultdict(float))
    for session in sessions:
        tracks = [x[0] for x in session]
        times = [x[1] for x in session]
        for i, src in enumerate(tracks):
            for d in range(1, window + 1):
                j = i + d
                if j >= len(tracks):
                    break
                dst = tracks[j]
                w = (0.1 + times[i]) * (0.1 + times[j]) / (d ** 0.8)
                graph[src][dst] += w
                graph[dst][src] += 0.1 * w
    out = {}
    for src, vals in graph.items():
        z = max(vals.values()) if vals else 1.0
        out[src] = {dst: val / z for dst, val in vals.items()}
    return out


class SkipGram(torch.nn.Module):
    def __init__(self, n, dim):
        super().__init__()
        self.a = torch.nn.Embedding(n, dim)
        self.b = torch.nn.Embedding(n, dim)
        torch.nn.init.normal_(self.a.weight, std=0.02)
        torch.nn.init.normal_(self.b.weight, std=0.02)

    def forward(self, x, y, neg):
        vx = self.a(x)
        vy = self.b(y)
        vn = self.b(neg)
        pos = torch.sum(vx * vy, dim=1)
        neg_score = torch.bmm(vn, vx.unsqueeze(2)).squeeze(2)
        return -(torch.nn.functional.logsigmoid(pos).mean() + torch.nn.functional.logsigmoid(-neg_score).mean())


def make_pairs(sessions, item_to_idx, window, max_pairs):
    pairs = []
    for session in sessions:
        ids = [item_to_idx[t] for t, v in session if t in item_to_idx and v >= 0.15]
        for i, src in enumerate(ids):
            left = max(0, i - window)
            right = min(len(ids), i + window + 1)
            for j in range(left, right):
                if i != j:
                    pairs.append((src, ids[j]))
    if len(pairs) > max_pairs:
        random.seed(31312)
        pairs = random.sample(pairs, max_pairs)
    return np.asarray(pairs, dtype=np.int64)


def item2vec(sessions, tracks, args):
    item_to_idx = {x: i for i, x in enumerate(tracks)}
    idx_to_item = np.asarray(tracks, dtype=np.int64)
    pairs = make_pairs(sessions, item_to_idx, args.window, args.max_pairs)
    if len(pairs) == 0:
        return {}

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"device={device} pairs={len(pairs)}")

    model = SkipGram(len(tracks), args.dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    counts = np.ones(len(tracks), dtype=np.float64)
    for session in sessions:
        for track, t in session:
            if track in item_to_idx:
                counts[item_to_idx[track]] += max(t, 0.05)
    probs = torch.tensor(counts ** 0.75 / np.sum(counts ** 0.75), dtype=torch.float32, device=device)

    p = torch.tensor(pairs, dtype=torch.long)
    steps = max(1, math.ceil(len(p) / args.batch_size))

    for epoch in range(args.epochs):
        order = torch.randperm(len(p))
        losses = []
        for step in tqdm(range(steps), leave=False):
            idx = order[step * args.batch_size:(step + 1) * args.batch_size]
            batch = p[idx].to(device)
            x = batch[:, 0]
            y = batch[:, 1]
            neg = torch.multinomial(probs, x.shape[0] * args.negatives, replacement=True).view(x.shape[0], args.negatives)
            loss = model(x, y, neg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print("epoch", epoch + 1, round(sum(losses) / max(len(losses), 1), 6))

    out = {}
    with torch.no_grad():
        emb = torch.nn.functional.normalize(model.a.weight.detach(), dim=1)
        for start in tqdm(range(0, len(tracks), args.neighbor_batch), leave=False):
            part = emb[start:start + args.neighbor_batch]
            scores = part @ emb.T
            cols = torch.arange(start, min(start + args.neighbor_batch, len(tracks)), device=scores.device)
            scores[torch.arange(scores.shape[0], device=scores.device), cols] = -1e9
            top = torch.topk(scores, k=min(args.neural_topk, len(tracks) - 1), dim=1).indices.cpu().numpy()
            for row, ids in enumerate(top):
                src = int(idx_to_item[start + row])
                out[src] = {int(idx_to_item[i]): rank for rank, i in enumerate(ids)}
    return out


def write(path, rows):
    with open(path, "w") as f:
        for item, recs in rows:
            f.write(json.dumps({"item_id": int(item), "recommendations": [int(x) for x in recs]}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--catalog", default="botify/data/tracks.json")
    parser.add_argument("--source", default="botify/data/sasrec_i2i.jsonl")
    parser.add_argument("--out", default="botify/data/ensemble_i2i.jsonl")
    parser.add_argument("--user-out", default="botify/data/ensemble_user.jsonl")
    parser.add_argument("--prior", type=float, default=60.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta-transition", type=float, default=0.025)
    parser.add_argument("--beta-neural", type=float, default=0.012)
    parser.add_argument("--beta-artist", type=float, default=0.004)
    parser.add_argument("--rank-scale", type=float, default=80.0)
    parser.add_argument("--topk", type=int, default=500)
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--negatives", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--max-pairs", type=int, default=8000000)
    parser.add_argument("--neighbor-batch", type=int, default=256)
    parser.add_argument("--neural-topk", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    random.seed(31312)
    np.random.seed(31312)
    torch.manual_seed(31312)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(31312)

    tracks, artists = read_catalog(args.catalog)
    source = read_recs(args.source)
    df = read_logs(args.logs)
    sessions = build_sessions(df)
    q, mean = quality(df, args.prior)
    graph = transition_graph(sessions, args.window)
    neural = item2vec(sessions, tracks, args)

    result = []
    for item, recs in source:
        src_artist = artists.get(item, "")
        scored = []
        for rank, rec in enumerate(recs):
            score = -rank / args.rank_scale
            score += args.alpha * q.get(rec, mean)
            score += args.beta_transition * graph.get(item, {}).get(rec, 0.0)
            nrank = neural.get(item, {}).get(rec)
            if nrank is not None:
                score += args.beta_neural / ((nrank + 1) ** 0.5)
            if src_artist and artists.get(rec, "") == src_artist:
                score += args.beta_artist
            scored.append((score, -rank, rec))
        scored.sort(reverse=True)
        result.append((item, [rec for _, _, rec in scored[:args.topk]]))

    write(args.out, result)
    Path(args.user_out).write_text("")
    print("tracks", len(tracks))
    print("sessions", len(sessions))
    print("quality_tracks", len(q))
    print("global_mean", round(mean, 6))
    print("transition_sources", len(graph))
    print("neural_sources", len(neural))
    print(args.out)


if __name__ == "__main__":
    main()
