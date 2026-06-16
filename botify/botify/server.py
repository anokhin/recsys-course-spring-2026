import redis
import json
import logging
import time
import atexit
from dataclasses import asdict
from datetime import datetime

from flask import Flask
from flask_restful import Resource, Api, abort, reqparse
from gevent.pywsgi import WSGIServer

from botify.data import DataLogger, Datum
from botify.experiment import Experiments, Treatment
from botify.recommenders.i2i import I2IRecommender
from botify.recommenders.random import Random
from botify.recommenders.session_gate_ranker import SessionAwareGateRanker
from botify.track import Catalog

root = logging.getLogger()
root.setLevel("INFO")

app = Flask(__name__)
app.config.from_file("config.json", load=json.load)
api = Api(app)

# Use plain redis-py clients explicitly so host/port/db come from config.
tracks_redis = redis.Redis(
    host=app.config["REDIS_TRACKS_HOST"],
    port=app.config["REDIS_TRACKS_PORT"],
    db=app.config["REDIS_TRACKS_DB"],
)

artists_redis = redis.Redis(
    host=app.config["REDIS_ARTIST_HOST"],
    port=app.config["REDIS_ARTIST_PORT"],
    db=app.config["REDIS_ARTIST_DB"],
)

listen_history_redis = redis.Redis(
    host=app.config["REDIS_LISTEN_HISTORY_HOST"],
    port=app.config["REDIS_LISTEN_HISTORY_PORT"],
    db=app.config["REDIS_LISTEN_HISTORY_DB"],
)

recommendations_lfm_redis = redis.Redis(
    host=app.config["REDIS_RECOMMENDATIONS_LFM_HOST"],
    port=app.config["REDIS_RECOMMENDATIONS_LFM_PORT"],
    db=app.config["REDIS_RECOMMENDATIONS_LFM_DB"],
)

recommendations_contextual_redis = redis.Redis(
    host=app.config["REDIS_RECOMMENDATIONS_SASREC_HOST"],
    port=app.config["REDIS_RECOMMENDATIONS_SASREC_PORT"],
    db=app.config["REDIS_RECOMMENDATIONS_SASREC_DB"],
)

recommendations_hstu_redis = redis.Redis(
    host=app.config["REDIS_RECOMMENDATIONS_HSTU_HOST"],
    port=app.config["REDIS_RECOMMENDATIONS_HSTU_PORT"],
    db=app.config["REDIS_RECOMMENDATIONS_HSTU_DB"],
)

data_logger = DataLogger(app)
atexit.register(data_logger.close)

catalog = Catalog(app).load(app.config["TRACKS_CATALOG"])
catalog.upload_tracks(tracks_redis)
catalog.upload_artists(artists_redis)

random_recommender = Random(tracks_redis)

catalog.upload_recommendations(
    recommendations_lfm_redis,
    "RECOMMENDATIONS_LFM_FILE_PATH",
    key_object="item_id",
    key_recommendations="recommendations",
)

lightfm_i2i_recommender = I2IRecommender(
    listen_history_redis,
    recommendations_lfm_redis,
    random_recommender,
)

catalog.upload_recommendations(
    recommendations_contextual_redis,
    "RECOMMENDATIONS_SASREC_FILE_PATH",
    key_object="item_id",
    key_recommendations="recommendations",
)

catalog.upload_recommendations(
    recommendations_hstu_redis,
    "RECOMMENDATIONS_HSTU_FILE_PATH",
    key_object="user",
    key_recommendations="tracks",
)

sasrec_i2i_recommender = I2IRecommender(
    listen_history_redis,
    recommendations_contextual_redis,
    random_recommender,
)

session_gate_ranker_recommender = SessionAwareGateRanker(
    model_path=app.config["SESSION_GATE_RANKER_MODEL_PATH"],
    recommendations_sasrec_redis=recommendations_contextual_redis,
    recommendations_lfm_redis=recommendations_lfm_redis,
    tracks_redis=tracks_redis,
    listen_history_redis=listen_history_redis,
    baseline_recommender=sasrec_i2i_recommender,
    fallback_recommender=random_recommender,
    min_prev_time=0.70,
    abs_threshold=0.72,
    margin=0.08,
)

parser = reqparse.RequestParser()
parser.add_argument("track", type=int, location="json", required=True)
parser.add_argument("time", type=float, location="json", required=True)

LISTEN_HISTORY_LIMIT = 10


def persist_user_listen_history(user: int, track: int, track_time: float):
    user_history_key = f"user:{user}:listens"
    history_entry = json.dumps({"track": int(track), "time": float(track_time)})
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
        abort(404, description="Track not found")


class NextTrack(Resource):
    def post(self, user: int):
        start = time.time()

        args = parser.parse_args()
        persist_user_listen_history(user, args.track, args.time)

        treatment = Experiments.HSTU.assign(user)

        if treatment == Treatment.C:
            recommender = sasrec_i2i_recommender
        elif treatment == Treatment.T1:
            recommender = session_gate_ranker_recommender
        else:
            recommender = random_recommender

        try:
            recommendation = recommender.recommend_next(user, args.track, args.time)
        except Exception:
            app.logger.exception(
                f"recommend_next failed for user={user}, track={args.track}, time={args.time}"
            )
            recommendation = None

        if recommendation is None:
            try:
                recommendation = sasrec_i2i_recommender.recommend_next(
                    user, args.track, args.time
                )
            except Exception:
                recommendation = None

        if recommendation is None:
            try:
                recommendation = random_recommender.recommend_next(
                    user, args.track, args.time
                )
            except Exception:
                recommendation = None

        if recommendation is None:
            recommendation = 0

        recommendation = int(recommendation)

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
            experiments={"HSTU": treatment.name},
        )
        return {"user": user, "track": recommendation}
class LastTrack(Resource):
    def post(self, user: int):
        start = time.time()
        args = parser.parse_args()
        persist_user_listen_history(user, args.track, args.time)

        treatment = Experiments.HSTU.assign(user)

        data_logger.log(
            "last",
            Datum(
                int(datetime.now().timestamp() * 1000),
                user,
                args.track,
                args.time,
                time.time() - start,
            ),
            experiments={"HSTU": treatment.name},
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