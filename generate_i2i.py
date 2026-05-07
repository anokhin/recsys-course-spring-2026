import json
import pickle
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("1. Загружаем...")
with open('botify/botify/model_reg.pkl', 'rb') as f: model_reg = pickle.load(f)
with open('botify/botify/model_clf.pkl', 'rb') as f: model_clf = pickle.load(f)
embeddings = np.load('botify/botify/embeddings.npy')
with open('botify/botify/feature_names.pkl', 'rb') as f: feature_names = pickle.load(f)
with open('botify/botify/track_popularity.pkl', 'rb') as f: track_popularity = pickle.load(f)

sasrec_i2i = {}
with open('botify/data/sasrec_i2i.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        sasrec_i2i[int(data['item_id'])] = [int(r) for r in data['recommendations']]

print(f"Загружено {len(sasrec_i2i)} I2I словарей")

print("\n2. Генерируем...")
improved_i2i = {}

for item_id in tqdm(list(sasrec_i2i.keys())[:15000]):
    if item_id >= len(embeddings): continue
    
    candidates = [c for c in sasrec_i2i.get(item_id, []) if c < len(embeddings)]
    if len(candidates) <= 1:
        improved_i2i[item_id] = candidates[:10]
        continue
    
    track_emb = embeddings[item_id].astype(np.float32)
    features_list = []
    
    for cand_id in candidates:
        cand_emb = embeddings[cand_id].astype(np.float32)
        cos_sim = float(np.dot(track_emb, cand_emb))
        
        feat = {name: 0.0 for name in feature_names}
        for name in feature_names:
            if 'cos_' in name:
                feat[name] = cos_sim
        
        cp = track_popularity.get(item_id, {})
        feat['current_play_count'] = np.log1p(cp.get('play_count', 0))
        feat['current_avg_time'] = cp.get('avg_listen_time', 0.5)
        feat['current_skip_rate'] = cp.get('skip_rate', 0.5)
        
        rp = track_popularity.get(cand_id, {})
        feat['rec_play_count'] = np.log1p(rp.get('play_count', 0))
        feat['rec_avg_time'] = rp.get('avg_listen_time', 0.5)
        feat['rec_skip_rate'] = rp.get('skip_rate', 0.5)
        
        features_list.append([feat[name] for name in feature_names])
    
    X = np.array(features_list, dtype=np.float32)
    time_pred = model_reg.predict(X)
    skip_pred = model_clf.predict_proba(X)[:, 1]
    ensemble_scores = time_pred * (1 - skip_pred)
    
    sorted_candidates = [c for _, c in sorted(zip(ensemble_scores, candidates), key=lambda x: x[0], reverse=True)]
    improved_i2i[item_id] = sorted_candidates[:10]

print(f"Сгенерировано для {len(improved_i2i)} треков")

print("\n3. Сохраняем...")
with open('botify/data/improved_i2i.jsonl', 'w') as f:
    for item_id, recs in improved_i2i.items():
        f.write(json.dumps({'item_id': item_id, 'recommendations': recs}) + '\n')

print("✅ Готово!")