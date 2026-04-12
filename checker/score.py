import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

MAX_SCORE = 28.0
METRIC = "mean_time_per_session"


def get_last_commit_time(pr_url: str, token: str) -> datetime:
    if requests is None:
        raise ImportError("pip install requests")
    parts = pr_url.rstrip("/").split("/")
    owner, repo, pr_number = parts[-4], parts[-3], parts[-1]
    headers = {"Authorization": f"token {token}"} if token else {}
    resp = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/commits",
        headers=headers, timeout=30
    )
    resp.raise_for_status()
    commits = resp.json()
    if not commits:
        raise ValueError(f"PR #{pr_number} не содержит коммитов")
    ts = commits[-1]["commit"]["committer"]["date"]
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def deadline_days(submit: datetime, deadline: datetime) -> int:
    return -1 if submit <= deadline else int((submit - deadline).total_seconds() // 86400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr-url", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--ab-result", required=True)
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN", ""))
    args = parser.parse_args()

    ab_path = Path(args.ab_result)
    if not ab_path.exists():
        print(f"ab_result.json не найден: {ab_path}")
        sys.exit(1)

    ab = json.load(open(ab_path))
    effects = ab.get("all_effects", [])
    key = next((e for e in effects if e["metric"] == METRIC), None)

    if key is None:
        print(f"Метрика '{METRIC}' не найдена в all_effects")
        sys.exit(1)

    beat = float(key["effect_pct"]) > 0
    significant = bool(key.get("significant", False))
    effect_pct = float(key["effect_pct"])

    deadline = datetime.fromisoformat(args.deadline)
    print("📡 Получаем время последнего коммита...")
    try:
        submit_time = get_last_commit_time(args.pr_url, args.token)
        print(f"   Коммит:  {submit_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"   Дедлайн: {deadline.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    except Exception as e:
        print(f"GitHub API: {e}")
        sys.exit(1)

    d = deadline_days(submit_time, deadline)
    penalty = 0.95 ** (1 + d)
    score = round(MAX_SCORE * penalty, 2) if beat else 0.0
    formula = f"28 × 0.95^(1+{d}) = {score:.2f}"

    print(f"""
╔══════════════════════════════════════════╗
║           ИТОГОВЫЙ БАЛЛ ДЗ2              ║
╠══════════════════════════════════════════╣
║  mean_time_per_session:                  ║
║    effect_pct: {effect_pct:+.2f}%{"":<22}║
║    significant: {"OK" if significant else "NO":<27}║
║    Победил контроль: {"OK" if beat else "NO":<22}║
╠══════════════════════════════════════════╣
║  Дедлайн (d): {"вовремя" if d == -1 else f"+{d} сут.":<28}║
║  Множитель:   {penalty:<28.4f}║
╠══════════════════════════════════════════╣
║  БАЛЛ: {score:.1f} / {MAX_SCORE:<34}║
║  {formula:<42}║
╚══════════════════════════════════════════╝
""")

    result = {
        "score": score,
        "max_score": MAX_SCORE,
        "formula": formula,
        "beat_control": beat,
        "significant": significant,
        "effect_pct": effect_pct,
        "d": d,
        "penalty": round(penalty, 6),
        "submit_time": submit_time.isoformat(),
        "deadline": deadline.isoformat(),
    }
    with open("score_result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("score_result.json")
    sys.exit(0 if beat else 1)


if __name__ == "__main__":
    main()
