"""
Offline training script: builds hybrid item-to-item recommendations
combining SasRec, LightFM, and content embeddings via a learned blender.

Run once locally:
    pip install -r train/requirements.txt
    python -m train.build_hybrid_i2i

Output: botify/data/hybrid_i2i.jsonl  (committed to repo, loaded by botify at startup).
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


SEED = 31312
TOP_K_OUT = 10                # how many recs per anchor in final file
CAND_PER_SOURCE = 30          # candidates per source before blending
MUTUAL_RANK = 5               # rank threshold for "mutual neighbour" positives
NEG_PER_POS = 4               # negatives per positive when training blender
MAX_PER_ARTIST = 1            # cap on tracks per artist in final top-10 (artist-diverse re-rank)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("hybrid_i2i")


def load_tracks(path: Path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    n = max(r["track"] for r in rows) + 1
    by_id = [None] * n
    for r in rows:
        by_id[r["track"]] = r
    return by_id


def load_i2i(path: Path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[int(r["item_id"])] = [int(x) for x in r["recommendations"]]
    return out


def build_text(track):
    if track is None:
        return ""
    parts = [
        track.get("title") or "",
        "by " + (track.get("artist") or ""),
    ]
    genres = track.get("genres") or []
    if genres:
        parts.append("genres: " + ", ".join(genres))
    if track.get("mood"):
        parts.append("mood: " + track["mood"])
    if track.get("year"):
        parts.append("year: " + str(track["year"]))
    artist_genres = track.get("artist_genres") or []
    if artist_genres:
        parts.append("artist genres: " + ", ".join(artist_genres))
    if track.get("artist_country"):
        parts.append("country: " + track["artist_country"])
    if track.get("summary"):
        parts.append(track["summary"])
    return ". ".join(parts)


def encode_texts(texts):
    """Embed track texts. Prefer sentence-transformers; fall back to TF-IDF+SVD."""
    try:
        from sentence_transformers import SentenceTransformer
        log.info("encoding %d texts with sentence-transformers/all-MiniLM-L6-v2", len(texts))
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb = model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return emb.astype(np.float32)
    except Exception as e:
        log.warning("sentence-transformers unavailable (%s); falling back to TF-IDF+SVD", e)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        vec = TfidfVectorizer(min_df=2, max_df=0.6, ngram_range=(1, 2), max_features=50000)
        X = vec.fit_transform(texts)
        svd = TruncatedSVD(n_components=256, random_state=SEED)
        emb = svd.fit_transform(X)
        emb = normalize(emb, axis=1).astype(np.float32)
        return emb


def content_topk(emb, k):
    """For every item, return indices of top-k most similar items (excluding self).
    Computed in chunks to bound memory."""
    n = emb.shape[0]
    out = np.zeros((n, k), dtype=np.int32)
    chunk = 1024
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = emb[start:end] @ emb.T
        for i in range(end - start):
            sims[i, start + i] = -1.0
        idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]
        for i in range(end - start):
            row_idx = idx[i]
            row_sims = sims[i, row_idx]
            order = np.argsort(-row_sims)
            out[start + i] = row_idx[order]
    return out


def make_pair_features(anchor, candidate, sasrec, lfm, content_idx, emb, tracks):
    rs = sasrec.get(anchor, [])
    rl = lfm.get(anchor, [])
    rc = content_idx[anchor].tolist()

    def rank(lst, item, default):
        try:
            return 1.0 / (1 + lst.index(item))
        except ValueError:
            return default

    rr_s = rank(rs, candidate, 0.0)
    rr_l = rank(rl, candidate, 0.0)
    rr_c = rank(rc, candidate, 0.0)
    in_s = float(candidate in rs)
    in_l = float(candidate in rl)
    in_c = float(candidate in rc)
    csim = float(emb[anchor] @ emb[candidate])
    a, c = tracks[anchor], tracks[candidate]
    same_artist = float(a is not None and c is not None and a.get("artist_id") == c.get("artist_id"))
    a_genres = set(a.get("artist_genres") or []) if a else set()
    c_genres = set(c.get("artist_genres") or []) if c else set()
    g_overlap = len(a_genres & c_genres) / max(1, len(a_genres | c_genres)) if (a_genres or c_genres) else 0.0
    a_mood = a.get("mood") if a else None
    c_mood = c.get("mood") if c else None
    same_mood = float(bool(a_mood) and a_mood == c_mood)
    def _year(t):
        y = t.get("year") if t else None
        try:
            return int(y) if y is not None else 2000
        except (TypeError, ValueError):
            return 2000
    year_diff = abs(_year(a) - _year(c))
    year_close = 1.0 / (1.0 + year_diff / 5.0)
    return [rr_s, rr_l, rr_c, in_s, in_l, in_c, csim, same_artist, g_overlap, same_mood, year_close]


FEATURE_NAMES = [
    "rr_sasrec", "rr_lfm", "rr_content",
    "in_sasrec", "in_lfm", "in_content",
    "content_sim", "same_artist", "genre_jaccard",
    "same_mood", "year_close",
]


def train_blender(sasrec, lfm, content_idx, emb, tracks, rng):
    """Weak-supervision: positives are mutual SasRec neighbours (a in top-MUTUAL_RANK of b
    AND b in top-MUTUAL_RANK of a) — pairs that are stably similar from both sides.
    Negatives are random pairs that are NOT mutual neighbours.
    """
    log.info("collecting blender training pairs (mutual SasRec neighbours)")
    sasrec_set_topk = {a: set(rs[:MUTUAL_RANK]) for a, rs in sasrec.items()}
    pos_pairs = []
    for a, rs in sasrec.items():
        for b in rs[:MUTUAL_RANK]:
            if b in sasrec_set_topk and a in sasrec_set_topk[b] and a != b:
                pos_pairs.append((a, b))
    log.info("positive pairs: %d", len(pos_pairs))

    anchors_with_recs = list(sasrec.keys())
    n_items = emb.shape[0]
    neg_pairs = []
    target_neg = len(pos_pairs) * NEG_PER_POS
    while len(neg_pairs) < target_neg:
        a = rng.choice(anchors_with_recs)
        b = int(rng.integers(0, n_items))
        if b == a:
            continue
        if b in sasrec_set_topk.get(a, set()):
            continue
        neg_pairs.append((a, b))

    log.info("building features: %d pos + %d neg", len(pos_pairs), len(neg_pairs))
    X, y = [], []
    for a, b in pos_pairs:
        X.append(make_pair_features(a, b, sasrec, lfm, content_idx, emb, tracks))
        y.append(1)
    for a, b in neg_pairs:
        X.append(make_pair_features(a, b, sasrec, lfm, content_idx, emb, tracks))
        y.append(0)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    log.info("fitting LogisticRegression blender on %d samples", len(y))
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    clf.fit(X, y)
    log.info("train accuracy: %.4f", clf.score(X, y))
    log.info("learned weights:")
    for name, w in zip(FEATURE_NAMES, clf.coef_[0]):
        log.info("  %-15s %+.4f", name, w)
    return clf


def artist_diverse_pick(cands_sorted, tracks, k, max_per_artist):
    """Greedy: walk through candidates by descending score, accept a track
    only if its artist hasn't been used `max_per_artist` times already.
    If we run out before k, relax the cap and continue.
    """
    picked = []
    artist_count = {}
    leftovers = []
    for c in cands_sorted:
        a = tracks[c].get("artist_id") if tracks[c] else -1
        if artist_count.get(a, 0) < max_per_artist:
            picked.append(c)
            artist_count[a] = artist_count.get(a, 0) + 1
            if len(picked) >= k:
                return picked
        else:
            leftovers.append(c)
    for c in leftovers:
        picked.append(c)
        if len(picked) >= k:
            break
    return picked[:k]


def build_recs(sasrec, lfm, content_idx, emb, tracks, clf):
    n_items = emb.shape[0]
    out = {}
    log.info("scoring + artist-diverse re-ranking for %d anchors", n_items)
    for anchor in range(n_items):
        rs = sasrec.get(anchor, [])
        rl = lfm.get(anchor, [])
        rc = content_idx[anchor].tolist()
        cands = []
        seen = set()
        for src in (rs[:CAND_PER_SOURCE], rl[:CAND_PER_SOURCE], rc[:CAND_PER_SOURCE]):
            for c in src:
                if c == anchor or c in seen or c >= n_items:
                    continue
                seen.add(c)
                cands.append(c)
        if not cands:
            out[anchor] = [c for c in rc if c != anchor][:TOP_K_OUT]
            continue
        feats = np.asarray(
            [make_pair_features(anchor, c, sasrec, lfm, content_idx, emb, tracks) for c in cands],
            dtype=np.float32,
        )
        scores = clf.predict_proba(feats)[:, 1]
        order = np.argsort(-scores)
        ranked = [cands[i] for i in order]
        picked = artist_diverse_pick(ranked, tracks, k=TOP_K_OUT, max_per_artist=MAX_PER_ARTIST)
        out[anchor] = picked
        if anchor % 2000 == 0:
            log.info("  anchor %d / %d", anchor, n_items)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", default="botify/data/tracks.json")
    p.add_argument("--sasrec", default="botify/data/sasrec_i2i.jsonl")
    p.add_argument("--lfm", default="botify/data/lightfm_i2i.jsonl")
    p.add_argument("--out", default="botify/data/hybrid_i2i.jsonl")
    p.add_argument("--embeddings-cache", default="train/embeddings.npy")
    args = p.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    tracks = load_tracks(Path(args.tracks))
    log.info("loaded %d tracks", len(tracks))
    sasrec = load_i2i(Path(args.sasrec))
    lfm = load_i2i(Path(args.lfm))
    log.info("loaded %d sasrec / %d lfm anchors", len(sasrec), len(lfm))

    cache = Path(args.embeddings_cache)
    if cache.exists():
        log.info("loading cached embeddings from %s", cache)
        emb = np.load(cache).astype(np.float32)
        if emb.shape[0] != len(tracks):
            log.warning("cache size mismatch (%d vs %d); recomputing", emb.shape[0], len(tracks))
            emb = None
    else:
        emb = None

    if emb is None:
        texts = [build_text(t) for t in tracks]
        emb = encode_texts(texts)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        log.info("saved embeddings to %s (shape %s)", cache, emb.shape)

    log.info("computing content top-%d for all items", CAND_PER_SOURCE)
    content_idx = content_topk(emb, k=CAND_PER_SOURCE)

    clf = train_blender(sasrec, lfm, content_idx, emb, tracks, rng)
    recs = build_recs(sasrec, lfm, content_idx, emb, tracks, clf)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item_id in sorted(recs.keys()):
            f.write(json.dumps({"item_id": item_id, "recommendations": recs[item_id]}) + "\n")
    log.info("wrote %d anchors → %s", len(recs), out_path)


if __name__ == "__main__":
    main()
