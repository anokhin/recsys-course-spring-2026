import argparse
import json
import sys

METRIC = "mean_time_per_session"
THRESHOLD = 10.0


def get_effect(ab_path: str) -> dict:
    try:
        data = json.load(open(ab_path))
        effects = data.get("all_effects", [])
        return next((e for e in effects if e["metric"] == METRIC), None)
    except Exception as e:
        print(f"Ошибка чтения {ab_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ab1", required=True)
    parser.add_argument("--ab2", required=True)
    parser.add_argument("--output", default="repro_result.json")
    args = parser.parse_args()

    e1 = get_effect(args.ab1)
    e2 = get_effect(args.ab2)

    if e1 is None or e2 is None:
        result = {"ok": False, "error": f"Метрика '{METRIC}' не найдена в ab_result.json"}
        json.dump(result, open(args.output, "w"), indent=2)
        print(f"{result['error']}")
        sys.exit(1)

    p1 = float(e1["effect_pct"])
    p2 = float(e2["effect_pct"])
    delta = abs(p1 - p2)
    sign_match = (p1 > 0) == (p2 > 0)
    ok = sign_match and delta <= THRESHOLD

    if not sign_match:
        verdict = f"Знаки не совпадают: run1={p1:+.2f}% run2={p2:+.2f}%"
    elif delta > THRESHOLD:
        verdict = f"Расхождение слишком большое: delta={delta:.2f}% > {THRESHOLD}% (порог)"
    else:
        verdict = f"Воспроизводимо: знаки совпадают, delta={delta:.2f}% ≤ {THRESHOLD}%"

    result = {
        "ok": ok,
        "run1_effect": round(p1, 2),
        "run2_effect": round(p2, 2),
        "delta": round(delta, 2),
        "threshold": THRESHOLD,
        "verdict": verdict,
    }

    print(f"  run1: effect={p1:+.2f}%  significant={e1['significant']}")
    print(f"  run2: effect={p2:+.2f}%  significant={e2['significant']}")
    print(f"  delta={delta:.2f}%  (порог={THRESHOLD}%)")
    print(f"  {verdict}")

    json.dump(result, open(args.output, "w"), indent=2, ensure_ascii=False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
