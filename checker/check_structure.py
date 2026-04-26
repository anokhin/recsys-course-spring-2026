import re
import sys
from pathlib import Path

REQUIRED_TARGETS = ["setup", "run", "clean"]
REQUIRED_VARIABLES = ["SEED", "EPISODES", "DATA_DIR"]


def check(repo_path: str) -> bool:
    repo = Path(repo_path).resolve()
    mf = repo / "Makefile"
    ok = True

    if not mf.exists():
        print(" Makefile не найден в корне репо")
        return False

    content = mf.read_text(encoding="utf-8", errors="ignore")

    for t in REQUIRED_TARGETS:
        if re.search(rf"^{t}\s*:", content, re.MULTILINE):
            print(f"Таргет '{t}'")
        else:
            print(f"Таргет '{t}' не найден")
            ok = False

    for v in REQUIRED_VARIABLES:
        if re.search(rf"^{v}\s*\?=", content, re.MULTILINE):
            print(f"Переменная '{v}' (?=)")
        elif re.search(rf"^{v}\s*=", content, re.MULTILINE):
            print(f"'{v}' есть, но без ?= — проверяющий не сможет переопределить")
        else:
            print(f"Переменная '{v}' не найдена")
            ok = False

    if "analyze_ab.py" in content and "--data" in content and "--output" in content:
        print("analyze_ab.py вызывается в make run")
    else:
        print("В make run не найден вызов: analyze_ab.py --data ... --output ...")
        ok = False

    if (repo / "analyze_ab.py").exists():
        print("analyze_ab.py найден в репо")
    else:
        print("analyze_ab.py не найден в корне репо")
        ok = False

    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python check_structure.py <path_to_repo>")
        sys.exit(1)
    sys.exit(0 if check(sys.argv[1]) else 1)
