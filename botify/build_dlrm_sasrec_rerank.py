import argparse
import json
from pathlib import Path


def load_i2i(path: Path):
    data = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            data[int(row["item_id"])] = [int(x) for x in row["recommendations"]]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sasrec", type=Path, default=Path("data/sasrec_i2i.jsonl"))
    parser.add_argument("--dlrm", type=Path, default=Path("data/dlrm_transition_i2i.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/dlrm_sasrec_rerank_i2i.jsonl"))
    parser.add_argument("--lambda-dlrm", type=float, default=0.006)
    args = parser.parse_args()

    sasrec = load_i2i(args.sasrec)
    dlrm = load_i2i(args.dlrm)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    changed_first = 0

    with args.output.open("w") as out:
        for item_id, sas_candidates in sasrec.items():
            dlrm_candidates = dlrm.get(item_id, [])
            dlrm_rank = {track: rank for rank, track in enumerate(dlrm_candidates)}
            missing_rank = len(dlrm_candidates) + 100

            original_first = sas_candidates[0] if sas_candidates else None
            scored = []
            for sas_rank, track in enumerate(sas_candidates):
                # SasRec remains dominant; DLRM only nudges candidates inside SasRec top list.
                score = sas_rank + args.lambda_dlrm * dlrm_rank.get(track, missing_rank)
                scored.append((score, sas_rank, track))

            reranked = [track for _, _, track in sorted(scored)]
            if original_first is not None and reranked and reranked[0] != original_first:
                changed_first += 1

            out.write(json.dumps({"item_id": int(item_id), "recommendations": reranked}) + "\n")
            written += 1

    print(f"written={written} changed_first={changed_first} output={args.output}")


if __name__ == "__main__":
    main()
