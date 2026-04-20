import json
import logging
import os
import pickle
from pathlib import Path

log = logging.getLogger(__name__)


def _artifact_dir() -> Path:
    return Path(os.environ.get("RERANKER_DIR", "./data/reranker"))


def load_booster():
    import lightgbm as lgb
    path = _artifact_dir() / "model.txt"
    if not path.exists():
        return None
    return lgb.Booster(model_file=str(path))


def load_meta():
    path = _artifact_dir() / "meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_item2vec():
    path = _artifact_dir() / "item2vec.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def load_track_meta(tracks_json_path: str):
    meta = {}
    with open(tracks_json_path) as f:
        for line in f:
            row = json.loads(line)
            meta[int(row["track"])] = {"artist": row["artist"]}
    return meta


def build_reranker(listen_history_redis, sasrec_redis, fallback, tracks_json_path):
    """
    Load artifacts and return an `LGBMReranker`. If artifacts are missing, return the
    fallback recommender so the service still starts. CI always has artifacts committed.
    """
    from .lgbm_reranker import LGBMReranker

    booster = load_booster()
    meta = load_meta()
    if booster is None or meta is None:
        log.warning("Reranker artifacts missing; falling back to SasRec-I2I in T1")
        return fallback

    track_meta = load_track_meta(tracks_json_path)
    item2vec = load_item2vec()

    return LGBMReranker(
        listen_history_redis=listen_history_redis,
        sasrec_redis=sasrec_redis,
        fallback=fallback,
        booster=booster,
        feature_order=meta["feature_order"],
        track_meta=track_meta,
        item2vec=item2vec,
        top_k=meta.get("top_k", 50),
    )
