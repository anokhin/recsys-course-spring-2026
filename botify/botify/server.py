import json
import logging
import os
import time
import atexit
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from flask import Flask
from flask_redis import Redis
from flask_restful import Resource, Api, abort, reqparse
from gevent.pywsgi import WSGIServer

from botify.data import DataLogger, Datum
from botify.experiment import Experiments, Treatment
from botify.recommenders.i2i import I2IRecommender
from botify.recommenders.random import Random
from botify.recommenders.session_blend import SessionBlendRecommender
from botify.track import Catalog

root = logging.getLogger()
root.setLevel("INFO")

app = Flask(__name__)
app.config.from_file(os.getenv("BOTIFY_CONFIG_PATH", "config.json"), load=json.load)
api = Api(app)

listen_history_redis = Redis(app, config_prefix="REDIS_LISTEN_HISTORY")

data_logger = DataLogger(app)
atexit.register(data_logger.close)


def load_recommendations(path, key_object, key_recommendations):
    result = {}
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            result[int(row[key_object])] = [int(track) for track in row[key_recommendations]]
    return result


catalog = Catalog(app).load(app.config["TRACKS_CATALOG"])
track_lookup = {track.track: track for track in catalog.tracks}
all_track_ids = [track.track for track in catalog.tracks]
lightfm_store = load_recommendations(app.config["RECOMMENDATIONS_LFM_FILE_PATH"], "item_id", "recommendations")
sasrec_store = load_recommendations(app.config["RECOMMENDATIONS_SASREC_FILE_PATH"], "item_id", "recommendations")
hstu_store = load_recommendations(app.config["RECOMMENDATIONS_HSTU_FILE_PATH"], "user", "tracks")

random_recommender = Random(all_track_ids)
lightfm_i2i_recommender = I2IRecommender(
    listen_history_redis.connection,
    lightfm_store,
    random_recommender,
)

sasrec_i2i_recommender = I2IRecommender(
    listen_history_redis.connection,
    sasrec_store,
    random_recommender,
)

session_blend_path = Path(app.config["SESSION_BLEND_MODEL_FILE_PATH"])
session_blend_recommender = (
    SessionBlendRecommender(
        listen_history_redis.connection,
        sasrec_store,
        lightfm_store,
        hstu_store,
        catalog,
        sasrec_i2i_recommender,
        session_blend_path,
    )
    if session_blend_path.exists()
    else None
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
        if track in track_lookup:
            return asdict(track_lookup[track])
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
            recommender = session_blend_recommender or sasrec_i2i_recommender
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
