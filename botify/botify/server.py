import json
import logging
import time
import atexit
from dataclasses import asdict
from datetime import datetime

import redis
from flask import Flask
from flask_restful import Resource, Api, abort, reqparse
from gevent.pywsgi import WSGIServer

from botify.data import DataLogger, Datum
from botify.experiment import Experiments, Treatment
from botify.recommenders.i2i import I2IRecommender
from botify.recommenders.random import Random
from botify.recommenders.learned_gate_ranker import LearnedGateRanker
from botify.recommenders.sticky_artist import StickyArtist
from botify.track import Catalog

root = logging.getLogger()
root.setLevel("INFO")

app = Flask(__name__)
app.config.from_file("config.json", load=json.load)
api = Api(app)

def _redis_from_config(prefix: str) -> redis.Redis:
    return redis.Redis(
        host=app.config[f"{prefix}_HOST"],
        port=app.config[f"{prefix}_PORT"],
        db=app.config[f"{prefix}_DB"],
    )


tracks_redis = _redis_from_config("REDIS_TRACKS")
artists_redis = _redis_from_config("REDIS_ARTIST")
listen_history_redis = _redis_from_config("REDIS_LISTEN_HISTORY")
recommendations_lfm_redis = _redis_from_config("REDIS_RECOMMENDATIONS_LFM")
recommendations_contextual_redis = _redis_from_config("REDIS_RECOMMENDATIONS_SASREC")
recommendations_hstu_redis = _redis_from_config("REDIS_RECOMMENDATIONS_HSTU")
session_state_redis = _redis_from_config("REDIS_SESSION_STATE")

data_logger = DataLogger(app)
atexit.register(data_logger.close)

catalog = Catalog(app).load(app.config["TRACKS_CATALOG"])
catalog.upload_tracks(tracks_redis)
catalog.upload_artists(artists_redis)

catalog.upload_recommendations(
    recommendations_lfm_redis,
    "RECOMMENDATIONS_LFM_FILE_PATH",
    key_object="item_id",
    key_recommendations="recommendations",
)
lightfm_i2i_recommender = I2IRecommender(
    listen_history_redis,
    recommendations_lfm_redis,
    Random(tracks_redis),
)

catalog.upload_recommendations(
    recommendations_contextual_redis,
    "RECOMMENDATIONS_SASREC_FILE_PATH",
    key_object="item_id",
    key_recommendations="recommendations",
)
sasrec_i2i_recommender = I2IRecommender(
    listen_history_redis,
    recommendations_contextual_redis,
    Random(tracks_redis),
)

catalog.upload_recommendations(
    recommendations_hstu_redis,
    "RECOMMENDATIONS_HSTU_FILE_PATH",
)

learned_gate_ranker = LearnedGateRanker(
    model_path=app.config["LEARNED_RANKER_MODEL_PATH"],
    meta_path=app.config["LEARNED_RANKER_META_PATH"],
    tracks_meta_path=app.config["TRACKS_CATALOG"],
    sasrec_redis=recommendations_contextual_redis,
    lightfm_redis=recommendations_lfm_redis,
    hstu_redis=recommendations_hstu_redis,
    listen_history_redis=listen_history_redis,
    baseline_recommender=sasrec_i2i_recommender,
    fallback_recommender=Random(tracks_redis),
)

parser = reqparse.RequestParser()
parser.add_argument("track", type=int, location="json", required=True)
parser.add_argument("time", type=float, location="json", required=True)

LISTEN_HISTORY_LIMIT = 10


def persist_user_listen_history(user: int, track: int, track_time: float):
    user_history_key = f"user:{user}:listens"
    history_entry = json.dumps({"track": track, "time": track_time})
    listen_history_redis.lpush(user_history_key, history_entry)
    listen_history_redis.ltrim(user_history_key, 0, LISTEN_HISTORY_LIMIT - 1)


class Hello(Resource):
    def get(self):
        return {
            "status": "alive",
            "message": "welcome to botify, the best toy music recommender",
        }


class Track(Resource):
    def get(self, track: int):
        data = tracks_redis.get(track)
        if data is not None:
            return asdict(catalog.from_bytes(data))
        else:
            abort(404, description="Track not found")


class NextTrack(Resource):
    def post(self, user: int):
        start = time.time()

        args = parser.parse_args()
        persist_user_listen_history(user, args.track, args.time)

        treatment = Experiments.HW2_RANKER.assign(user)

        if treatment == Treatment.C:
            recommender = sasrec_i2i_recommender
        elif treatment == Treatment.T1:
            recommender = learned_gate_ranker
        else:
            recommender = Random(tracks_redis)

        recommendation = None
        try:
            recommendation = recommender.recommend_next(user, args.track, args.time)
        except Exception:
            app.logger.exception("primary recommender failed")
        if recommendation is None:
            try:
                recommendation = sasrec_i2i_recommender.recommend_next(user, args.track, args.time)
            except Exception:
                app.logger.exception("sasrec fallback failed")
        if recommendation is None:
            try:
                recommendation = lightfm_i2i_recommender.recommend_next(user, args.track, args.time)
            except Exception:
                app.logger.exception("lightfm fallback failed")
        try:
            recommendation = int(recommendation) if recommendation is not None else int(args.track)
        except (TypeError, ValueError):
            recommendation = int(args.track)

        data_logger.log(
            "next",
            Datum(
                int(datetime.now().timestamp() * 1000),
                user,
                args.track,
                args.time,
                time.time() - start,
                recommendation,
            ),
        )
        return {"user": user, "track": recommendation}


class LastTrack(Resource):
    def post(self, user: int):
        start = time.time()
        args = parser.parse_args()
        persist_user_listen_history(user, args.track, args.time)

        data_logger.log(
            "last",
            Datum(
                int(datetime.now().timestamp() * 1000),
                user,
                args.track,
                args.time,
                time.time() - start,
            ),
        )
        return {"user": user}


api.add_resource(Hello, "/")
api.add_resource(Track, "/track/<int:track>")
api.add_resource(NextTrack, "/next/<int:user>")
api.add_resource(LastTrack, "/last/<int:user>")

app.logger.info("Botify service started")

if __name__ == "__main__":
    http_server = WSGIServer(("", 5001), app)
    http_server.serve_forever()
