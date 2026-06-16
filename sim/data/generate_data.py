import json
import shutil
import random
from pathlib import Path

import numpy as np

random.seed(31312)
np.random.seed(31312)

SIM_DIR = Path(__file__).parent
BOTIFY_TRACKS = Path(__file__).parent.parent.parent / "botify" / "data" / "tracks.json"
EMBED_DIM = 64
WEIGHT_NOISE = 0.1
WEIGHT_INTEREST_GENRE = 0.7
WEIGHT_INTEREST_ARTIST = 0.3
N_USERS = 10000


def load_tracks():
    tracks = []
    with open(BOTIFY_TRACKS) as f:
        for line in f:
            t = json.loads(line)
            tracks.append({
                "track": int(t["track"]),
                "artist": t["artist"],
                "genres": set(t.get("genres", [])),
                "artist_id": int(t.get("artist_id", 0)),
            })
    return tracks


def build_embeddings(tracks):
    all_genres = sorted(set(g for t in tracks for g in t["genres"]))
    all_artists = sorted(set(t["artist"] for t in tracks))

    print(f"Genres: {len(all_genres)}, Artists: {len(all_artists)}")

    genre_vecs = {}
    for g in all_genres:
        v = np.random.randn(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        genre_vecs[g] = v

    artist_vecs = {}
    for a in all_artists:
        v = np.random.randn(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        artist_vecs[a] = v

    embeddings = np.zeros((len(tracks), EMBED_DIM), dtype=np.float32)

    for i, t in enumerate(tracks):
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        n_genres = len(t["genres"])
        if n_genres > 0:
            for g in t["genres"]:
                if g in genre_vecs:
                    vec += genre_vecs[g]
            vec /= n_genres
        vec += WEIGHT_NOISE * np.random.randn(EMBED_DIM).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        embeddings[i] = vec

    return embeddings, all_genres, all_artists


def build_users(tracks, embeddings, all_genres, all_artists):
    track_by_genre = {}
    for t in tracks:
        for g in t["genres"]:
            track_by_genre.setdefault(g, []).append(t["track"])

    track_by_artist = {}
    for t in tracks:
        track_by_artist.setdefault(t["artist"], []).append(t["track"])

    all_track_ids = [t["track"] for t in tracks]
    genre_list = sorted(all_genres)
    artist_list = sorted(all_artists)

    users = []
    for uid in range(N_USERS):
        n_genres = random.randint(2, 6)
        preferred_genres = set(random.sample(genre_list, min(n_genres, len(genre_list))))

        n_artists = random.randint(2, 8)
        preferred_artists = set(random.sample(artist_list, min(n_artists, len(artist_list))))

        interest_tracks = []
        for g in preferred_genres:
            candidates = track_by_genre.get(g, [])
            if candidates:
                interest_tracks.extend(random.sample(candidates, min(3, len(candidates))))
        for a in preferred_artists:
            candidates = track_by_artist.get(a, [])
            if candidates:
                interest_tracks.extend(random.sample(candidates, min(2, len(candidates))))

        if not interest_tracks:
            interest_tracks = random.sample(all_track_ids, min(10, len(all_track_ids)))
        else:
            interest_tracks = list(set(interest_tracks))

        consume_bias = round(random.uniform(3.0, 7.0), 2)
        consume_sharpness = round(random.uniform(0.5, 2.0), 2)
        session_budget = random.randint(3, 10)
        artist_discount_gamma = round(random.uniform(0.6, 0.95), 2)

        users.append({
            "user": uid,
            "interests": interest_tracks,
            "interest_neighbours": 10,
            "consume_bias": consume_bias,
            "consume_sharpness": consume_sharpness,
            "session_budget": session_budget,
            "artist_discount_gamma": artist_discount_gamma,
        })

        if (uid + 1) % 2000 == 0:
            print(f"  Generated {uid + 1}/{N_USERS} users")

    return users


def main():
    print("Loading tracks from botify...")
    tracks = load_tracks()
    print(f"Loaded {len(tracks)} tracks")

    print("\nBuilding embeddings...")
    embeddings, all_genres, all_artists = build_embeddings(tracks)
    emb_path = SIM_DIR / "embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings.npy: {embeddings.shape} ({emb_path.stat().st_size // 1024} KB)")

    print("\nCopying tracks.json...")
    tracks_path = SIM_DIR / "tracks.json"
    shutil.copy(BOTIFY_TRACKS, tracks_path)
    print(f"Copied tracks.json ({tracks_path.stat().st_size // 1024} KB)")

    print("\nGenerating users...")
    users = build_users(tracks, embeddings, all_genres, all_artists)
    users_path = SIM_DIR / "users.json"
    with open(users_path, "w") as f:
        for u in users:
            f.write(json.dumps(u) + "\n")
    print(f"Saved users.json: {len(users)} users ({users_path.stat().st_size // 1024} KB)")

    print("\nDone! All sim data generated.")


if __name__ == "__main__":
    main()
