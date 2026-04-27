import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


INPUT_PATH = Path("botify/data/tracks.json")
OUTPUT_PATH = Path("botify/data/content_i2i.jsonl")
FIELDS = (
    "title",
    "alternative_title",
    "artist",
    "alternative_artist",
    "genres",
    "mood",
    "summary",
    "artist_country",
    "artist_genres",
    "artist_genre",
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
MIN_TOKEN_LENGTH = 3
MAX_DF_RATIO = 0.06
MAX_DF_ABSOLUTE = 500
MAX_TERMS_PER_DOC = 40
TOP_K = 100
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "are",
    "was",
    "were",
    "his",
    "her",
    "she",
    "him",
    "you",
    "your",
    "about",
    "into",
    "their",
    "them",
    "they",
    "then",
    "than",
    "when",
    "what",
    "where",
    "who",
    "how",
    "has",
    "have",
    "had",
    "not",
    "but",
    "all",
    "one",
    "its",
    "itself",
    "song",
    "music",
    "lyrics",
    "track",
}


def iter_text(value):
    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from iter_text(item)
        return
    yield str(value)


def tokenize(record):
    tokens = []
    for field in FIELDS:
        for text in iter_text(record.get(field)):
            for token in TOKEN_PATTERN.findall(text.lower()):
                if len(token) >= MIN_TOKEN_LENGTH and token not in STOPWORDS:
                    tokens.append(token)
    return tokens


def read_records():
    records = []
    with INPUT_PATH.open() as source:
        for line in source:
            record = json.loads(line)
            records.append(record)
    records.sort(key=lambda item: int(item["track"]))
    return records


def build_vectors(records):
    counters = []
    doc_freq = Counter()
    for record in records:
        counter = Counter(tokenize(record))
        counters.append(counter)
        doc_freq.update(counter.keys())

    doc_count = len(records)
    max_df = max(2, min(MAX_DF_ABSOLUTE, int(doc_count * MAX_DF_RATIO)))
    valid_terms = {
        term for term, freq in doc_freq.items()
        if 2 <= freq <= max_df
    }
    idf = {
        term: math.log((1 + doc_count) / (1 + doc_freq[term])) + 1
        for term in valid_terms
    }

    vectors = []
    for counter in counters:
        weighted = []
        for term, count in counter.items():
            if term not in idf:
                continue
            weight = (1 + math.log(count)) * idf[term]
            weighted.append((term, weight))

        weighted.sort(key=lambda item: (-item[1], item[0]))
        weighted = weighted[:MAX_TERMS_PER_DOC]
        norm = math.sqrt(sum(weight * weight for _, weight in weighted))
        if norm:
            vectors.append([(term, weight / norm) for term, weight in weighted])
        else:
            vectors.append([])
    return vectors


def build_index(vectors):
    index = defaultdict(list)
    for doc_index, vector in enumerate(vectors):
        for term, weight in vector:
            index[term].append((doc_index, weight))
    return index


def top_similar(doc_index, vectors, index, track_ids):
    scores = defaultdict(float)
    for term, weight in vectors[doc_index]:
        for other_index, other_weight in index[term]:
            if other_index != doc_index:
                scores[other_index] += weight * other_weight

    ranked = sorted(scores.items(), key=lambda item: (-item[1], track_ids[item[0]]))
    recommendations = []
    used = {track_ids[doc_index]}

    for other_index, _ in ranked:
        track = track_ids[other_index]
        if track not in used:
            recommendations.append(track)
            used.add(track)
        if len(recommendations) == TOP_K:
            return recommendations

    for track in track_ids:
        if track not in used:
            recommendations.append(track)
            used.add(track)
        if len(recommendations) == TOP_K:
            break

    return recommendations


def main():
    records = read_records()
    track_ids = [int(record["track"]) for record in records]
    vectors = build_vectors(records)
    index = build_index(vectors)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as target:
        for doc_index, track in enumerate(track_ids):
            output = {
                "item_id": track,
                "recommendations": top_similar(doc_index, vectors, index, track_ids),
            }
            target.write(json.dumps(output, separators=(",", ":")) + "\n")

    print(f"wrote {OUTPUT_PATH}")
    print(f"tracks={len(track_ids)}")
    print(f"top_k={TOP_K}")


if __name__ == "__main__":
    main()
