"""
Compute mean_time_per_session per RERANKER arm from collected botify logs.
Sessions delimited by 'last' events; session_time = sum of 'next' time + 'last' time.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict


def load_events(paths):
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def sessions_by_arm(events):
    per_user = defaultdict(list)
    for e in events:
        if e.get("message") in ("next", "last"):
            per_user[e["user"]].append(e)

    sessions = []
    for user, evs in per_user.items():
        evs.sort(key=lambda r: r["timestamp"])
        buf = []
        for e in evs:
            buf.append(e)
            if e["message"] == "last":
                arm = None
                total = 0.0
                for x in buf:
                    if x["message"] == "next":
                        a = x.get("experiments", {}).get("RERANKER")
                        if a is not None:
                            arm = a
                    total += float(x.get("time", 0.0))
                if arm is not None:
                    sessions.append((arm, total))
                buf = []
    return sessions


def welch(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    ma = sum(a) / na
    mb = sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("nan"), float("nan")
    t = (mb - ma) / se
    return t, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="glob for data.json files")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.logs, recursive=True))
    if not paths:
        raise SystemExit(f"No logs matched {args.logs}")
    sess = sessions_by_arm(load_events(paths))

    by_arm = defaultdict(list)
    for arm, tot in sess:
        by_arm[arm].append(tot)

    arms = sorted(by_arm)
    print(f"{'arm':<6} {'n_sessions':>12} {'mean_time':>10} {'std':>10}")
    for a in arms:
        xs = by_arm[a]
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
        print(f"{a:<6} {len(xs):>12d} {m:>10.4f} {v**0.5:>10.4f}")

    if "C" in by_arm and "T1" in by_arm:
        t, se = welch(by_arm["C"], by_arm["T1"])
        delta = sum(by_arm["T1"]) / len(by_arm["T1"]) - sum(by_arm["C"]) / len(by_arm["C"])
        print(f"\nDelta(T1 - C) = {delta:+.4f}   Welch-t = {t:+.3f}   SE = {se:.4f}")


if __name__ == "__main__":
    main()
