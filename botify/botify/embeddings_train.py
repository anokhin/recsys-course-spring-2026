import json
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

with open('../data/tracks.json', 'r', encoding='utf-8') as f:
    tracks_data = [json.loads(line) for line in f if line.strip()]

corpus = []
track_ids = []

for track in tracks_data:
    tid = int(track['track'])
    parts = [
        str(track.get('title', '')),
        str(track.get('artist', '')),
        " ".join(track.get('genres', [])),
        str(track.get('mood', '')),
        str(track.get('summary', '')),
    ]

    text = " ".join([p.strip() for p in parts if p.strip()])
    corpus.append(text)
    track_ids.append(tid)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.7,
    sublinear_tf=True
)
tfidf_matrix = vectorizer.fit_transform(corpus)

n_components = min(128, tfidf_matrix.shape[1] - 1)
svd = TruncatedSVD(n_components=n_components, random_state=35)
embeddings = svd.fit_transform(tfidf_matrix)

embeddings = normalize(embeddings, norm='l2', axis=1)

embeddings_dict = {}
for i, tid in enumerate(track_ids):
    embeddings_dict[tid] = embeddings[i]

output_file = '../data/embeddings.joblib'

joblib.dump(embeddings_dict, output_file, compress=3, protocol=4)