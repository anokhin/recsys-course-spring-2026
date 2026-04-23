import json
import logging
import time
import atexit
from dataclasses import asdict
from datetime import datetime

from flask import Flask
from flask_redis import Redis
from flask_restful import Resource, Api, abort, reqparse
from gevent.pywsgi import WSGIServer

from botify.data import DataLogger, Datum
from botify.experiment import Experiments, Treatment
from botify.recommenders.i2i import I2IRecommender
from botify.recommenders.random import Random
from botify.recommenders.session_semantic import SessionSemanticRecommender
from botify.track import Catalog

root = logging.getLogger()
root.setLevel("INFO")

app = Flask(__name__)
app.config.from_file("config.json", load=json.load)
api = Api(app)

tracks_redis = Redis(app, config_prefix="REDIS_TRACKS")
artists_redis = Redis(app, config_prefix="REDIS_ARTIST")
listen_history_redis = Redis(app, config_prefix="REDIS_LISTEN_HISTORY")
recommendations_contextual_redis = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_SASREC")
recommendations_lfm_redis = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_LFM")
recommendations_hstu_redis = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_HSTU")

data_logger = DataLogger(app)
atexit.register(data_logger.close)

catalog = Catalog(app).load(app.config["TRACKS_CATALOG"])
catalog.upload_tracks(tracks_redis.connection)
catalog.upload_artists(artists_redis.connection)

random_recommender = Random(tracks_redis.connection)

catalog.upload_recommendations(
    recommendations_contextual_redis.connection,
    "RECOMMENDATIONS_SASREC_FILE_PATH",
    key_object="item_id",
    key_recommendations="recommendations",
)
catalog.upload_recommendations(
    recommendations_lfm_redis.connection,
    "RECOMMENDATIONS_LFM_FILE_PATH",
    key_object="item_id",
    key_recommendations="recommendations",
)
catalog.upload_recommendations(
    recommendations_hstu_redis.connection,
    "RECOMMENDATIONS_HSTU_FILE_PATH",
    key_object="user",
    key_recommendations="tracks",
)

sasrec_i2i_recommender = I2IRecommender(
    listen_history_redis.connection,
    recommendations_contextual_redis.connection,
    random_recommender,
)
session_semantic_recommender = SessionSemanticRecommender(
    listen_history_redis.connection,
    catalog,
    app.config["SESSION_SEMANTIC_EMBEDDINGS_FILE_PATH"],
    sasrec_i2i_recommender,
    i2i_redis=recommendations_contextual_redis.connection,
    lfm_i2i_redis=recommendations_lfm_redis.connection,
    hstu_redis=recommendations_hstu_redis.connection,
    artist_penalty=0.22,
    session_profile_weight=0.78,
    prototype_weight=0.22,
    hstu_prior_weight=0.0,
    i2i_bonus=0.07,
    lfm_bonus=0.0,
    hstu_bonus=0.0,
    semantic_gate=0.14,
    min_margin=0.010,
)

parser = reqparse.RequestParser()
parser.add_argument("track", type=int, location="json", required=True)
parser.add_argument("time", type=float, location="json", required=True)

LISTEN_HISTORY_LIMIT = 10


def persist_user_listen_history(user: int, track: int, track_time: float):
    user_history_key = f"user:{user}:listens"
    history_entry = json.dumps({"track": track, "time": track_time})
    listen_history_redis.connection.lpush(user_history_key, history_entry)
    listen_history_redis.connection.ltrim(user_history_key, 0, LISTEN_HISTORY_LIMIT - 1)


class Hello(Resource):
    def get(self):
        return {
            "status": "alive",
            "message": "welcome to botify, the best toy music recommender",
        }


class Track(Resource):
    def get(self, track: int):
        data = tracks_redis.connection.get(track)
        if data is not None:
            return asdict(catalog.from_bytes(data))
        else:
            abort(404, description="Track not found")


class NextTrack(Resource):
    def post(self, user: int):
        start = time.time()

        args = parser.parse_args()
        persist_user_listen_history(user, args.track, args.time)

        treatment = Experiments.SESSION_SEMANTIC.assign(user)

        if treatment == Treatment.C:
            recommender = sasrec_i2i_recommender
        else:
            session_semantic_recommender.observe(user, args.track, args.time)
            recommender = session_semantic_recommender

        recommendation = recommender.recommend_next(user, args.track, args.time)

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
        if Experiments.SESSION_SEMANTIC.assign(user) != Treatment.C:
            session_semantic_recommender.observe(user, args.track, args.time)
            session_semantic_recommender.finish_session(user)
        data_logger.log(
            "last",
            Datum(
                int(datetime.now().timestamp() * 1000),
                user,
                args.track,
                args.time,
                time.time() - start,
            )
        )
        return {"user": user}


api.add_resource(Hello, "/")
api.add_resource(Track, "/track/<int:track>")
api.add_resource(NextTrack, "/next/<int:user>")
api.add_resource(LastTrack, "/last/<int:user>")

app.logger.info(f"Botify service stared")

if __name__ == "__main__":
    http_server = WSGIServer(("", 5001), app)
    http_server.serve_forever()
