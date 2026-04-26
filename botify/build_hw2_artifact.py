import argparse
from pathlib import Path

from botify.personalization import (
    build_recommendation_rows,
    fit_recommendation_artifacts,
    read_interactions,
    write_recommendation_rows,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    interactions = read_interactions(Path(args.data))
    artifacts = fit_recommendation_artifacts(interactions)
    rows = build_recommendation_rows(artifacts, top_k=args.top_k)
    write_recommendation_rows(rows, Path(args.output))
    print("users={0} prev_tracks={1} tracks={2} interactions={3}".format(
        interactions["user"].nunique(),
        len(artifacts.prev_track_index),
        len(artifacts.track_index),
        len(interactions),
    ))
    print("saved={0}".format(args.output))


if __name__ == "__main__":
    main()
