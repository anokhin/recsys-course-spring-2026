"""
Train item-item similarity model using shared training data.
Generates recommendations in the same format as sasrec_i2i.jsonl
"""
import json
import pickle
import glob
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, lil_matrix


def load_logs(data_dir):
    """Load all JSON log files from directory"""
    files = glob.glob(f"{data_dir}/**/data.json*", recursive=True)
    interactions = []
    print(f"Found {len(files)} log files")
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("message") == "next":
                        interactions.append((
                            entry["user"],
                            entry["track"],
                            entry["time"]
                        ))
                except:
                    pass
    print(f"Loaded {len(interactions)} interactions")
    return interactions


def build_item_similarity(interactions, n_items=15001):
    """Build item-item cosine similarity matrix from interactions"""
    # Build user-item matrix
    users = set(u for u, t, _ in interactions)
    user_to_idx = {u: i for i, u in enumerate(users)}
    n_users = len(users)

    print(f"Building {n_users} x {n_items} interaction matrix...")
    matrix = lil_matrix((n_users, n_items), dtype=np.float32)
    for user, track, time in interactions:
        if track < n_items:
            matrix[user_to_idx[user], track] += time

    # Convert to CSR and compute item-item similarity
    matrix = matrix.tocsr()
    print("Computing item-item cosine similarity...")
    item_item = cosine_similarity(matrix.T, dense_output=False)
    
    return item_item


def generate_recommendations(similarity, n_items=15001, top_k=10):
    """Generate top-k recommendations for each item"""
    output = []
    for i in range(n_items):
        row = similarity.getrow(i).toarray().flatten()
        # Set self-similarity to -inf
        row[i] = -1.0
        top_indices = np.argsort(row)[::-1][:top_k]
        top_tracks = [int(idx) for idx in top_indices if row[idx] > 0]
        output.append({
            "item_id": i,
            "recommendations": top_tracks[:top_k]
        })
    return output


def main():
    # Load training data
    all_data = (
        load_logs("botify/data/training_2") +
        load_logs("data_from_rec1") +
        load_logs("data_from_rec2")
    )

    # Build similarity
    similarity = build_item_similarity(all_data)

    # Generate recommendations
    recommendations = generate_recommendations(similarity)

    # Save
    out_path = "botify/data/trained_i2i.jsonl"
    with open(out_path, 'w') as f:
        for rec in recommendations:
            f.write(json.dumps(rec) + '\n')

    print(f"Saved {len(recommendations)} item recommendations to {out_path}")


if __name__ == "__main__":
    main()