# Hybrid i2i training

Offline build of `botify/data/hybrid_i2i.jsonl` — committed artefact loaded by botify.

```
python -m venv .train-venv && source .train-venv/bin/activate
pip install -r train/requirements.txt
python -m train.build_hybrid_i2i
```

Inputs (already in repo): `botify/data/tracks.json`, `sasrec_i2i.jsonl`, `lightfm_i2i.jsonl`.
Output: `botify/data/hybrid_i2i.jsonl` (≈15K anchors, top-10 each).
Intermediate cache: `train/embeddings.npy` (~25 MB, regenerated only if missing).

Deterministic for a fixed `SEED=31312`.
