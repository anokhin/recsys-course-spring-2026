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
from botify.recommenders.online_ranker import OnlineRankerRecommender
from botify.recommenders.random import Random
from botify.track import Catalog

root = logging.getLogger()
root.setLevel("INFO")

app = Flask(__name__)
app.config.from_file("config.json", load=json.load)
api = Api(app)

tracks_redis = Redis(app, config_prefix="REDIS_TRACKS")
artists_redis = Redis(app, config_prefix="REDIS_ARTIST")
listen_history_redis = Redis(app, config_prefix="REDIS_LISTEN_HISTORY")
recommendations_lfm_redis = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_LFM")
recommendations_contextual_redis = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_SASREC")
recommendations_dlrm_sasrec_rerank_redis = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_DLRM_SASREC_RERANK")

data_logger = DataLogger(app)
atexit.register(data_logger.close)

catalog = Catalog(app).load(app.config["TRACKS_CATALOG"])
catalog.upload_tracks(tracks_redis.connection)
catalog.upload_artists(artists_redis.connection)

random_recommender = Random(tracks_redis.connection)

catalog.upload_recommendations(
    recommendations_lfm_redis.connection,
    "RECOMMENDATIONS_LFM_FILE_PATH",
    key_object="item_id",
    key_recommendations="recommendations",
)
lightfm_i2i_recommender = I2IRecommender(
    listen_history_redis.connection,
    recommendations_lfm_redis.connection,
    random_recommender,
)

catalog.upload_recommendations(
    recommendations_contextual_redis.connection,
    "RECOMMENDATIONS_SASREC_FILE_PATH",
    key_object="item_id",
    key_recommendations="recommendations",
)

catalog.upload_recommendations(
    recommendations_dlrm_sasrec_rerank_redis.connection,
    "RECOMMENDATIONS_DLRM_SASREC_RERANK_FILE_PATH",
    key_object="item_id",
    key_recommendations="recommendations",
)

sasrec_i2i_recommender = I2IRecommender(
    listen_history_redis.connection,
    recommendations_contextual_redis.connection,
    random_recommender,
)

dlrm_sasrec_rerank_i2i_recommender = I2IRecommender(
    listen_history_redis.connection,
    recommendations_dlrm_sasrec_rerank_redis.connection,
    random_recommender,
)

online_ranker_recommender = OnlineRankerRecommender(
    listen_history_redis.connection,
    recommendations_contextual_redis.connection,
    recommendations_lfm_redis.connection,
    recommendations_dlrm_sasrec_rerank_redis.connection,
    app.config["ONLINE_RANKER_MODEL_PATH"],
    app.config["TRACKS_CATALOG"],
    dlrm_sasrec_rerank_i2i_recommender,
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

        treatment = Experiments.HOMEWORK2.assign(user)

        if treatment == Treatment.C:
            recommender = sasrec_i2i_recommender
        elif treatment == Treatment.T1:
            recommender = online_ranker_recommender
        else:
            recommender = sasrec_i2i_recommender

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

app.logger.info(f"Botify service stared")

if __name__ == "__main__":
    http_server = WSGIServer(("", 5001), app)
    http_server.serve_forever()
