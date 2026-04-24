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
from botify.recommenders.diverse_i2i import DiverseI2IRecommender
from botify.recommenders.i2i import I2IRecommender
from botify.recommenders.random import Random
from botify.track import Catalog

root = logging.getLogger()
root.setLevel("INFO")

app = Flask(__name__)
app.config.from_file("config.json", load=json.load)
api = Api(app)

tracks_redis = Redis(app, config_prefix="REDIS_TRACKS")
listen_history_redis = Redis(app, config_prefix="REDIS_LISTEN_HISTORY")
recommendations_sasrec_redis = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_SASREC")
recommendations_content_redis = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_CONTENT")

data_logger = DataLogger(app)
atexit.register(data_logger.close)

catalog = Catalog(app).load(app.config["TRACKS_CATALOG"])
catalog.upload_tracks(tracks_redis.connection)

random_recommender = Random(tracks_redis.connection)

catalog.upload_recommendations(
    recommendations_sasrec_redis.connection,
    "RECOMMENDATIONS_SASREC_FILE_PATH",
    key_object="item_id",
    key_recommendations="recommendations",
)

sasrec_i2i_recommender = I2IRecommender(
    listen_history_redis.connection,
    recommendations_sasrec_redis.connection,
    random_recommender,
)

catalog.upload_recommendations(
    recommendations_content_redis.connection,
    "RECOMMENDATIONS_CONTENT_FILE_PATH",
    key_object="item_id",
    key_recommendations="recommendations",
)

content_i2i_recommender = DiverseI2IRecommender(
    listen_history_redis.connection,
    recommendations_content_redis.connection,
    random_recommender,
    catalog,
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

        treatment = Experiments.HSTU.assign(user)

        if treatment == Treatment.C:
            recommender = sasrec_i2i_recommender
        elif treatment == Treatment.T1:
            recommender = content_i2i_recommender
        else:
            recommender = random_recommender

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
