import argparse
import glob
import json
import math
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


Session = namedtuple("Session", ["first_track", "pairs"])


def read_events(data_dir: Path) -> list:
    paths = (
        glob.glob(str(data_dir / "*/data.json"))
        or glob.glob(str(data_dir / "**/data.json"), recursive=True)
    )
    if not paths:
        raise FileNotFoundError(f"Не найдено data.json в {data_dir}")
    events = []
    for p in sorted(paths):
        with open(p) as f:
            for line in f:
                events.append(json.loads(line))
    return events


def sessionize(events: list) -> list:
    events.sort(key=lambda e: (e["user"], e["timestamp"]))

    sessions = []
    current_user = None
    buf = []
    for ev in events:
        if ev["user"] != current_user:
            if buf:
                sessions.extend(_flush_user(buf))
            current_user = ev["user"]
            buf = [ev]
        else:
            buf.append(ev)
    if buf:
        sessions.extend(_flush_user(buf))
    return sessions


def _flush_user(user_events: list) -> list:
    sessions = []
    session_tracks = []
    for ev in user_events:
        track = int(ev["track"])
        time = float(ev["time"])
        session_tracks.append((track, time))
        if ev.get("message") == "last":
            if len(session_tracks) >= 2:
                first_track = session_tracks[0][0]
                pairs = session_tracks[1:]
                sessions.append(Session(first_track, pairs))
            session_tracks = []
    return sessions


def build_training_pairs(sessions: list, min_time: float = 0.0):
    first_list = []
    played_list = []
    time_list = []
    for s in sessions:
        for played_track, time in s.pairs:
            if time < min_time:
                continue
            first_list.append(s.first_track)
            played_list.append(played_track)
            time_list.append(time)
    return (
        np.asarray(first_list, dtype=np.int64),
        np.asarray(played_list, dtype=np.int64),
        np.asarray(time_list, dtype=np.float32),
    )


def train(
    n_tracks: int,
    firsts: np.ndarray,
    playeds: np.ndarray,
    times: np.ndarray,
    d: int = 64,
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 5e-3,
    l2: float = 1e-4,
    device: str = "cpu",
    seed: int = 31312,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)

    clipped = np.clip(times, 0.01, 0.99)
    targets = np.log(clipped / (1.0 - clipped)).astype(np.float32)
    weights = times.astype(np.float32)

    n = len(firsts)
    perm = np.random.permutation(n)
    val_size = max(1, n // 20)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    def to_tensors(idx):
        return (
            torch.from_numpy(firsts[idx]),
            torch.from_numpy(playeds[idx]),
            torch.from_numpy(targets[idx]),
            torch.from_numpy(weights[idx]),
        )

    tr_first, tr_played, tr_target, tr_weight = to_tensors(train_idx)
    val_first, val_played, val_target, val_weight = to_tensors(val_idx)

    dataset = TensorDataset(tr_first, tr_played, tr_target, tr_weight)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    E = nn.Embedding(n_tracks, d)
    nn.init.normal_(E.weight, mean=0.0, std=1.0 / math.sqrt(d))
    E.to(device)

    opt = optim.Adam(E.parameters(), lr=lr, weight_decay=l2)

    best_val = float("inf")
    best_weights = E.weight.detach().cpu().numpy().copy()
    patience = 3
    stale = 0

    for epoch in range(1, epochs + 1):
        E.train()
        total_loss = 0.0
        total_w = 0.0
        for fb, pb, yb, wb in loader:
            fb = fb.to(device)
            pb = pb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            pred = (E(fb) * E(pb)).sum(dim=-1)
            sq = (pred - yb) ** 2
            loss = (wb * sq).sum() / (wb.sum() + 1e-8)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float((wb * sq).sum().item())
            total_w += float(wb.sum().item())

        E.eval()
        with torch.no_grad():
            vpred = (E(val_first.to(device)) * E(val_played.to(device))).sum(dim=-1)
            vsq = (vpred - val_target.to(device)) ** 2
            vw = val_weight.to(device)
            val_loss = float((vw * vsq).sum().item() / (vw.sum().item() + 1e-8))
        train_loss = total_loss / (total_w + 1e-8)
        print(f"epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss + 1e-5 < best_val:
            best_val = val_loss
            best_weights = E.weight.detach().cpu().numpy().copy()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {epoch}")
                break

    return best_weights


def save_jsonl(embeddings: np.ndarray, out_path: Path, precision: int = 5):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for track_id in range(embeddings.shape[0]):
            vec = [round(float(x), precision) for x in embeddings[track_id]]
            f.write(
                json.dumps({"item_id": int(track_id), "embedding": vec}) + "\n"
            )
    print(f"saved {embeddings.shape[0]} embeddings to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("./data"))
    parser.add_argument("--tracks", type=Path, default=Path("./botify/data/tracks.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("./botify/data/session_embeddings.jsonl"))
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--min-time", type=float, default=0.0,
                        help="фильтр пар: time >= min_time (0 = все)")
    parser.add_argument("--seed", type=int, default=31312)
    args = parser.parse_args()

    with open(args.tracks) as f:
        n_tracks = sum(1 for _ in f)
    print(f"n_tracks = {n_tracks}")

    events = read_events(args.data)
    print(f"events = {len(events)}")
    sessions = sessionize(events)
    print(f"sessions = {len(sessions)}")

    firsts, playeds, times = build_training_pairs(sessions, min_time=args.min_time)
    print(f"training pairs = {len(firsts)} "
          f"(time_mean={times.mean():.3f}, >0 share={(times > 0).mean():.3f})")

    embeddings = train(
        n_tracks=n_tracks,
        firsts=firsts,
        playeds=playeds,
        times=times,
        d=args.dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
    )
    save_jsonl(embeddings, args.output)


if __name__ == "__main__":
    main()
