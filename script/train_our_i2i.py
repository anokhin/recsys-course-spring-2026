"""Train custom item-to-item recommendations from allowed botify/data sources.

Strategy:
  1. Build PPMI co-occurrence over user listen-lists (hstu + user_ml).
  2. TruncatedSVD over PPMI -> collaborative item embeddings.
  3. Content embeddings from tracks.json (artist / genres / mood / year).
  4. Concat normalized embeddings, kNN by cosine, MMR-rerank by artist.
  5. Write {"item_id", "recommendations"} jsonl in the same format as sasrec_i2i.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def load_user_sessions(*paths: Path) -> list[list[int]]:
    sessions: list[list[int]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                tracks = [int(t) for t in row.get("tracks", [])]
                if len(tracks) >= 2:
                    sessions.append(tracks)
    return sessions


def load_i2i_pseudo_sessions(*paths: Path) -> list[list[int]]:
    sessions: list[list[int]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                anchor = int(row["item_id"])
                neighbours = [int(t) for t in row.get("recommendations", [])]
                if not neighbours:
                    continue
                sessions.append([anchor, *neighbours])
    return sessions


def load_i2i_neighbours(path: Path) -> dict[int, list[int]]:
    table: dict[int, list[int]] = {}
    if not path.exists():
        return table
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            table[int(row["item_id"])] = [
                int(t) for t in row.get("recommendations", [])
            ]
    return table


def load_tracks_metadata(path: Path) -> list[dict]:
    tracks: list[dict] = []
    with path.open() as f:
        for line in f:
            tracks.append(json.loads(line))
    tracks.sort(key=lambda r: int(r["track"]))
    return tracks


def build_cooccurrence(
    sessions: list[list[int]], n_items: int, window: int
) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for tracks in sessions:
        m = len(tracks)
        for i, anchor in enumerate(tracks):
            if anchor < 0 or anchor >= n_items:
                continue
            j_lo = max(0, i - window)
            j_hi = min(m, i + window + 1)
            for j in range(j_lo, j_hi):
                if j == i:
                    continue
                neighbour = tracks[j]
                if neighbour < 0 or neighbour >= n_items:
                    continue
                distance = abs(j - i)
                weight = 1.0 / distance
                rows.append(anchor)
                cols.append(neighbour)
                data.append(weight)
    coo = sparse.coo_matrix(
        (data, (rows, cols)), shape=(n_items, n_items), dtype=np.float32
    )
    return coo.tocsr()


def to_ppmi(cooc: sparse.csr_matrix, smoothing: float = 0.75) -> sparse.csr_matrix:
    row_sum = np.asarray(cooc.sum(axis=1)).flatten()
    col_sum = np.asarray(cooc.sum(axis=0)).flatten() ** smoothing
    total = float(col_sum.sum()) + 1e-9
    cooc = cooc.tocoo()
    data = cooc.data * total / (
        row_sum[cooc.row] * col_sum[cooc.col] + 1e-9
    )
    data = np.log(np.maximum(data, 1.0))
    keep = data > 0
    return sparse.coo_matrix(
        (data[keep], (cooc.row[keep], cooc.col[keep])),
        shape=cooc.shape,
        dtype=np.float32,
    ).tocsr()


def collab_embeddings(
    ppmi: sparse.csr_matrix, n_components: int, seed: int
) -> np.ndarray:
    n_components = min(n_components, min(ppmi.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    emb = svd.fit_transform(ppmi).astype(np.float32)
    return emb


def item2vec_embeddings(
    sessions: list[list[int]],
    n_items: int,
    vector_size: int,
    window: int,
    epochs: int,
    negative: int,
    seed: int,
) -> np.ndarray:
    from gensim.models import Word2Vec

    tokenized = [[str(t) for t in tracks] for tracks in sessions]
    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=1,
        negative=negative,
        epochs=epochs,
        workers=4,
        seed=seed,
    )
    emb = np.zeros((n_items, vector_size), dtype=np.float32)
    for token in model.wv.key_to_index:
        try:
            idx = int(token)
        except ValueError:
            continue
        if 0 <= idx < n_items:
            emb[idx] = model.wv[token]
    return emb


def content_embeddings(
    tracks: list[dict],
    n_items: int,
    n_components: int,
    top_artists: int,
    top_genres: int,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    artist_counter: Counter = Counter()
    genre_counter: Counter = Counter()
    mood_counter: Counter = Counter()
    for track in tracks:
        artist_counter[track.get("artist_id", -1)] += 1
        for genre in track.get("genres", []) or []:
            genre_counter[genre] += 1
        mood = track.get("mood")
        if mood:
            mood_counter[mood] += 1

    top_artist_ids = {a for a, _ in artist_counter.most_common(top_artists)}
    top_genre_set = {g for g, _ in genre_counter.most_common(top_genres)}
    mood_index = {m: i for i, (m, _) in enumerate(mood_counter.most_common())}

    artist_idx = {a: i for i, a in enumerate(sorted(top_artist_ids))}
    genre_idx = {g: i for i, g in enumerate(sorted(top_genre_set))}

    n_artist = len(artist_idx) + 1
    n_genre = len(genre_idx)
    n_mood = len(mood_index)
    n_year = 1

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    artist_of_track: list[int] = [-1] * n_items

    def parse_year(value) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        digits = ""
        for ch in str(value):
            if ch.isdigit():
                digits += ch
                if len(digits) == 4:
                    break
        if len(digits) == 4:
            return float(digits)
        return 0.0

    years = [parse_year(t.get("year")) for t in tracks]
    year_arr = np.array(years, dtype=np.float32)
    valid_years = year_arr[year_arr > 0]
    if valid_years.size > 0:
        year_min, year_max = float(valid_years.min()), float(valid_years.max())
    else:
        year_min, year_max = 0.0, 1.0
    year_range = max(year_max - year_min, 1.0)

    for track in tracks:
        idx = int(track["track"])
        if idx >= n_items:
            continue
        artist_id = track.get("artist_id", -1)
        artist_of_track[idx] = artist_id
        offset = 0
        a_pos = artist_idx.get(artist_id, len(artist_idx))
        rows.append(idx)
        cols.append(offset + a_pos)
        data.append(1.0)
        offset += n_artist

        for genre in track.get("genres", []) or []:
            if genre in genre_idx:
                rows.append(idx)
                cols.append(offset + genre_idx[genre])
                data.append(1.0)
        offset += n_genre

        mood = track.get("mood")
        if mood and mood in mood_index:
            rows.append(idx)
            cols.append(offset + mood_index[mood])
            data.append(1.0)
        offset += n_mood

        track_year = parse_year(track.get("year"))
        year_norm = (track_year - year_min) / year_range if track_year > 0 else 0.0
        rows.append(idx)
        cols.append(offset)
        data.append(float(year_norm))
        offset += n_year

    n_features = n_artist + n_genre + n_mood + n_year
    matrix = sparse.coo_matrix(
        (data, (rows, cols)), shape=(n_items, n_features), dtype=np.float32
    ).tocsr()

    n_components = max(1, min(n_components, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    emb = svd.fit_transform(matrix).astype(np.float32)
    return emb, artist_of_track


def stack_embeddings(
    collab: np.ndarray, content: np.ndarray, alpha: float
) -> np.ndarray:
    collab_n = normalize(collab) if collab.shape[1] > 0 else collab
    content_n = normalize(content) if content.shape[1] > 0 else content
    return np.hstack(
        [collab_n * float(alpha), content_n * float(1.0 - alpha)]
    ).astype(np.float32)


def topk_candidates(
    embeddings: np.ndarray, top_k: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    n = embeddings.shape[0]
    norm = normalize(embeddings)
    indices = np.empty((n, top_k), dtype=np.int32)
    scores = np.empty((n, top_k), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        sims = norm[start:end] @ norm.T
        for local_i, global_i in enumerate(range(start, end)):
            sims[local_i, global_i] = -np.inf
        part = np.argpartition(-sims, kth=top_k, axis=1)[:, :top_k]
        for row in range(end - start):
            row_scores = sims[row, part[row]]
            order = np.argsort(-row_scores)
            indices[start + row] = part[row][order]
            scores[start + row] = row_scores[order]
    return indices, scores


def fuse_candidates(
    item_idx: int,
    our_indices: np.ndarray,
    our_scores: np.ndarray,
    extra_neighbours: list[int],
    embeddings_norm: np.ndarray,
    fuse_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine our top-K with an external i2i top-K via reciprocal-rank fusion.

    Score is a convex combination of normalized cosine similarity (our embedding)
    and reciprocal rank from the external list. Self is filtered out.
    """
    fused: dict[int, float] = {}
    for rank, candidate in enumerate(our_indices.tolist()):
        if candidate == item_idx:
            continue
        score = float(our_scores[rank])
        fused[candidate] = fused.get(candidate, 0.0) + (1.0 - fuse_weight) * score

    if extra_neighbours:
        anchor_vec = embeddings_norm[item_idx]
        for rank, candidate in enumerate(extra_neighbours):
            if candidate == item_idx or candidate < 0:
                continue
            extra_score = 1.0 / (1.0 + rank)
            cosine = float(np.dot(anchor_vec, embeddings_norm[candidate]))
            fused[candidate] = fused.get(candidate, 0.0) + fuse_weight * (
                0.5 * extra_score + 0.5 * cosine
            )

    if not fused:
        return our_indices, our_scores

    items = np.fromiter(fused.keys(), dtype=np.int32)
    scores = np.fromiter(fused.values(), dtype=np.float32)
    order = np.argsort(-scores)
    return items[order], scores[order]


def mmr_rerank(
    item_idx: int,
    candidates: np.ndarray,
    cand_scores: np.ndarray,
    embeddings_norm: np.ndarray,
    artist_of_track: list[int],
    top_n: int,
    diversity_lambda: float,
) -> list[int]:
    selected: list[int] = []
    selected_artists: set[int] = set()
    chosen_mask = np.zeros(candidates.shape[0], dtype=bool)
    src_artist = artist_of_track[item_idx] if item_idx < len(artist_of_track) else -1
    if src_artist >= 0:
        selected_artists.add(src_artist)

    for _ in range(top_n):
        best_idx = -1
        best_score = -np.inf
        for k in range(candidates.shape[0]):
            if chosen_mask[k]:
                continue
            cand_track = int(candidates[k])
            cand_artist = (
                artist_of_track[cand_track] if cand_track < len(artist_of_track) else -1
            )
            relevance = float(cand_scores[k])
            artist_penalty = 1.0 if cand_artist in selected_artists else 0.0
            score = diversity_lambda * relevance - (1.0 - diversity_lambda) * artist_penalty
            if score > best_score:
                best_score = score
                best_idx = k
        if best_idx < 0:
            break
        chosen_mask[best_idx] = True
        cand_track = int(candidates[best_idx])
        selected.append(cand_track)
        cand_artist = (
            artist_of_track[cand_track] if cand_track < len(artist_of_track) else -1
        )
        if cand_artist >= 0:
            selected_artists.add(cand_artist)
    return selected


def write_jsonl(path: Path, n_items: int, final_recs: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item_id in range(n_items):
            recs = final_recs[item_id]
            if not recs:
                continue
            f.write(
                json.dumps({"item_id": item_id, "recommendations": recs}) + "\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=[
            "botify/data/hstu_recommendations.json",
            "botify/data/user_ml_recommendations.jsonl",
        ],
    )
    parser.add_argument(
        "--i2i-graphs",
        nargs="*",
        default=[
            "botify/data/lightfm_i2i.jsonl",
            "botify/data/sasrec_i2i.jsonl",
        ],
        help="Extra i2i jsonl files used as pseudo sessions (anchor + neighbours)",
    )
    parser.add_argument(
        "--i2i-graph-repeat",
        type=int,
        default=5,
        help="How many times to repeat i2i pseudo sessions in the corpus.",
    )
    parser.add_argument("--tracks", default="botify/data/tracks.json")
    parser.add_argument("--output", default="botify/data/session2vec_i2i.jsonl")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--collab-dim", type=int, default=64)
    parser.add_argument("--content-dim", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--top-candidates", type=int, default=50)
    parser.add_argument("--top-final", type=int, default=10)
    parser.add_argument("--mmr-lambda", type=float, default=0.9)
    parser.add_argument("--top-artists", type=int, default=200)
    parser.add_argument("--top-genres", type=int, default=50)
    parser.add_argument("--ppmi-smoothing", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=31312)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--collab-method",
        choices=["ppmi", "item2vec"],
        default="item2vec",
    )
    parser.add_argument("--item2vec-epochs", type=int, default=20)
    parser.add_argument("--item2vec-negative", type=int, default=10)
    parser.add_argument(
        "--fuse-i2i",
        default=None,
        help="Optional path to an external i2i jsonl whose neighbours are fused "
        "with our kNN candidates before MMR (e.g. lightfm).",
    )
    parser.add_argument(
        "--fuse-weight",
        type=float,
        default=0.5,
        help="Weight of external i2i in the rank fusion (0=our only, 1=external only).",
    )
    parser.add_argument(
        "--fuse-topk",
        type=int,
        default=20,
        help="How many neighbours from the external i2i list to pull per item.",
    )
    args = parser.parse_args()

    print(f"Loading tracks metadata from {args.tracks}")
    tracks = load_tracks_metadata(Path(args.tracks))
    n_items = max(int(t["track"]) for t in tracks) + 1
    print(f"  n_items = {n_items}")

    sessions = load_user_sessions(*[Path(p) for p in args.sessions])
    print(f"  loaded {len(sessions)} user sessions")
    if args.i2i_graphs:
        i2i_sessions = load_i2i_pseudo_sessions(*[Path(p) for p in args.i2i_graphs])
        for _ in range(args.i2i_graph_repeat):
            sessions.extend(i2i_sessions)
        print(
            f"  added {len(i2i_sessions)} i2i pseudo sessions "
            f"(x{args.i2i_graph_repeat})"
        )

    if args.collab_method == "ppmi":
        print("Building PPMI co-occurrence")
        cooc = build_cooccurrence(sessions, n_items=n_items, window=args.window)
        ppmi = to_ppmi(cooc, smoothing=args.ppmi_smoothing)
        print(f"  ppmi nnz = {ppmi.nnz}")

        print(f"Training collaborative SVD ({args.collab_dim}d)")
        collab = collab_embeddings(ppmi, args.collab_dim, seed=args.seed)
    else:
        print(
            f"Training Item2Vec ({args.collab_dim}d, window={args.window}, "
            f"epochs={args.item2vec_epochs})"
        )
        collab = item2vec_embeddings(
            sessions,
            n_items=n_items,
            vector_size=args.collab_dim,
            window=args.window,
            epochs=args.item2vec_epochs,
            negative=args.item2vec_negative,
            seed=args.seed,
        )
    print(f"  collab shape = {collab.shape}")

    print(f"Training content SVD ({args.content_dim}d)")
    content, artist_of_track = content_embeddings(
        tracks,
        n_items=n_items,
        n_components=args.content_dim,
        top_artists=args.top_artists,
        top_genres=args.top_genres,
        seed=args.seed,
    )
    print(f"  content shape = {content.shape}")

    embeddings = stack_embeddings(collab, content, alpha=args.alpha)
    print(f"  combined shape = {embeddings.shape}")

    print(
        f"Computing top-{args.top_candidates} kNN candidates "
        f"(batch_size={args.batch_size})"
    )
    cand_idx, cand_scores = topk_candidates(
        embeddings, top_k=args.top_candidates, batch_size=args.batch_size
    )

    norm_emb = normalize(embeddings)
    extra_neighbours: dict[int, list[int]] = {}
    if args.fuse_i2i:
        print(
            f"Fusing with external i2i {args.fuse_i2i} "
            f"(weight={args.fuse_weight}, topk={args.fuse_topk})"
        )
        extra_neighbours = load_i2i_neighbours(Path(args.fuse_i2i))
        print(f"  external entries = {len(extra_neighbours)}")

    print(f"MMR rerank to top-{args.top_final} (lambda={args.mmr_lambda})")
    final_recs: list[list[int]] = [[] for _ in range(n_items)]
    for item_idx in range(n_items):
        if extra_neighbours:
            extra = extra_neighbours.get(item_idx, [])[: args.fuse_topk]
            cand_items, cand_scores_local = fuse_candidates(
                item_idx=item_idx,
                our_indices=cand_idx[item_idx],
                our_scores=cand_scores[item_idx],
                extra_neighbours=extra,
                embeddings_norm=norm_emb,
                fuse_weight=args.fuse_weight,
            )
        else:
            cand_items = cand_idx[item_idx]
            cand_scores_local = cand_scores[item_idx]

        recs = mmr_rerank(
            item_idx=item_idx,
            candidates=cand_items,
            cand_scores=cand_scores_local,
            embeddings_norm=norm_emb,
            artist_of_track=artist_of_track,
            top_n=args.top_final,
            diversity_lambda=args.mmr_lambda,
        )
        final_recs[item_idx] = recs

    coverage = sum(1 for r in final_recs if r)
    print(f"Wrote recs for {coverage}/{n_items} items")

    write_jsonl(Path(args.output), n_items, final_recs)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
