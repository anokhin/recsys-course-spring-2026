import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def track_text(track):
    # добавляем повтор title для усиления значимости
    title = track.get("title") or ""
    values = [
        title,
        title, 
        track.get("mood") or "",
        track.get("artist_genre") or "",
        track.get("artist_country") or "",
        " ".join(track.get("genres") or []),
        " ".join(track.get("artist_genres") or []),
        track.get("summary") or "",
    ]
    return " ".join(str(value) for value in values)


def read_tracks(path):
    with open(path) as tracks_file:
        tracks = [json.loads(line) for line in tracks_file]
    return sorted(tracks, key=lambda track: int(track["track"]))


def build_recommendations(tracks, neighbours):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=2,
        max_df=0.6,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    vectors = vectorizer.fit_transform(track_text(track) for track in tracks)
    index = NearestNeighbors(
        n_neighbors=neighbours + 1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    index.fit(vectors)
    distances, indices = index.kneighbors(vectors, return_distance=True)

    for track, track_neighbours, dists in zip(tracks, indices, distances):
        track_id = int(track["track"])
      
        recommendations = []
        for item_id, distance in zip(track_neighbours, dists):
            item_id = int(item_id)
            if item_id != track_id and distance < 0.95:
                recommendations.append(item_id)
            if len(recommendations) >= neighbours:
                break
        yield {"item_id": track_id, "recommendations": recommendations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", default="botify/data/tracks.json")
    parser.add_argument("--output", default="botify/data/content_i2i.jsonl")
    parser.add_argument("--neighbours", type=int, default=50)
    parser.add_argument("--version", action="store_true") 
    args = parser.parse_args()

   
    if args.version:
        print("BUILDER_VERSION=2.1.0")
        return

    tracks = read_tracks(args.tracks)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    
    total = 0
    empty = 0
    
    with open(output, "w") as output_file:
        for row in build_recommendations(tracks, args.neighbours):
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1
            if not row["recommendations"]:
                empty += 1
    
    # Статистика
    print(f"Обработано: {total}, пустых: {empty} ({empty/total*100:.1f}%)")


if __name__ == "__main__":
    main()