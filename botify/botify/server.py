import atexit
import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime

import numpy as np
from flask import Flask
from flask_redis import Redis
from flask_restful import Resource, Api, abort, reqparse
from gevent.pywsgi import WSGIServer

from botify.data import DataLogger, Datum
from botify.experiment import Experiments, Treatment
from botify.recommenders.i2i import I2IRecommender
from botify.recommenders.ranked_i2i import RankedI2IRecommender
from botify.recommenders.random import Random
from botify.recommenders.sticky_artist import StickyArtist
from botify.track import Catalog

root = logging.getLogger()
root.setLevel("INFO")

app = Flask(__name__)
app.config.from_file("config.json", load=json.load)
api = Api(app)

tracks_redis = Redis(app, config_prefix="REDIS_TRACKS")
artists_redis = Redis(app, config_prefix="REDIS_ARTIST")
recommendations_sasrec_redis = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_SASREC")
user_history_redis = Redis(app, config_prefix="REDIS_USER_HISTORY")

data_logger = DataLogger(app)
atexit.register(data_logger.close)

catalog = Catalog(app).load(app.config["TRACKS_CATALOG"])
catalog.upload_tracks(tracks_redis.connection)
catalog.upload_artists(artists_redis.connection)


def _maybe_upload_recommendations(redis_conn, config_key: str) -> bool:
    path = app.config.get(config_key)
    if not path or not os.path.exists(path):
        app.logger.warning(
            "Recommendations file for %s not found (%s); skipping upload",
            config_key,
            path,
        )
        return False
    catalog.upload_recommendations(
        redis_conn,
        config_key,
        key_object="item_id",
        key_recommendations="recommendations",
    )
    return True


HAS_SASREC = _maybe_upload_recommendations(
    recommendations_sasrec_redis.connection, "RECOMMENDATIONS_SASREC_FILE_PATH"
)

CONTEXT_EMB = None
TRACK_EMB = None
context_emb_path = app.config.get("CONTEXT_EMB_PATH", "./data/context_emb.npy")
track_emb_path = app.config.get("TRACK_EMB_PATH", "./data/track_emb.npy")
if os.path.exists(context_emb_path) and os.path.exists(track_emb_path):
    CONTEXT_EMB = np.load(context_emb_path)
    TRACK_EMB = np.load(track_emb_path)
    app.logger.info(
        "Loaded ranker embeddings context=%s track=%s",
        CONTEXT_EMB.shape,
        TRACK_EMB.shape,
    )
HAS_RANKER = CONTEXT_EMB is not None and TRACK_EMB is not None

parser = reqparse.RequestParser()
parser.add_argument("track", type=int, location="json", required=True)
parser.add_argument("time", type=float, location="json", required=True)


class Hello(Resource):
    def get(self):
        return {
            "status": "alive",
            "message": "welcome to botify, the best toy music recommender",
            "has_sasrec": HAS_SASREC,
            "has_ranker": HAS_RANKER,
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

        fallback = Random(tracks_redis.connection)
        sticky = StickyArtist(
            tracks_redis.connection, artists_redis.connection, catalog
        )

        if not (HAS_SASREC and HAS_RANKER):
            recommender = sticky
        else:
            treatment = Experiments.HW2.assign(user)
            if treatment == Treatment.T1:
                recommender = RankedI2IRecommender(
                    recommendations_sasrec_redis.connection,
                    user_history_redis.connection,
                    catalog,
                    fallback,
                    CONTEXT_EMB,
                    TRACK_EMB,
                )
            else:
                recommender = I2IRecommender(
                    recommendations_sasrec_redis.connection,
                    user_history_redis.connection,
                    catalog,
                    fallback,
                )

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

app.logger.info(
    "Botify service started (sasrec=%s, ranker=%s)", HAS_SASREC, HAS_RANKER
)

if __name__ == "__main__":
    http_server = WSGIServer(("", 5001), app)
    http_server.serve_forever()
