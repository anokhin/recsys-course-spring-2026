import re


def unindent(text: str):
    return " ".join(line.lstrip() for line in text.splitlines())


def normalize(text):
    return re.sub(r"[\r\n]+", " ", text) if isinstance(text, str) else "unknown"


def parse_list_response(text: str, tag: str = None):
    items = []

    for line in text.split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            try:
                number, item = line.split(".", 1)
                number = int(number.strip())
                item = item.strip()
                if item:
                    items.append(
                        {
                            "item": item,
                            "id": number,
                            "tag": tag,
                        }
                    )
            except ValueError:
                continue

    return items


def retry(func, retries=10):
    attempt = 0
    while attempt < retries:
        try:
            return func()
        except Exception:
            attempt += 1
            if attempt >= retries:
                raise
