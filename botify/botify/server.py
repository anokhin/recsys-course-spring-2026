import atexit
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime

from flask import Flask
from flask_redis import Redis
from flask_restful import Resource, Api, abort, reqparse
from gevent.pywsgi import WSGIServer

from botify.data import DataLogger, Datum
from botify.experiment import Experiments, Treatment
from botify.ml import load_artifacts
from botify.recommenders.i2i import I2IRecommender
from botify.recommenders.ml_reranker import MLReranker
from botify.recommenders.random import Random
from botify.track import Catalog

LISTEN_HISTORY_LIMIT = 10

root = logging.getLogger()
root.setLevel("INFO")

app = Flask(__name__)
app.config.from_file("config.json", load=json.load)
api = Api(app)

tracks_redis = Redis(app, config_prefix="REDIS_TRACKS")
artists_redis = Redis(app, config_prefix="REDIS_ARTIST")
listen_history_redis = Redis(app, config_prefix="REDIS_LISTEN_HISTORY")
recommendations_lfm_redis = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_LFM")
recommendations_sasrec_redis = Redis(app, config_prefix="REDIS_RECOMMENDATIONS_SASREC")

data_logger = DataLogger(app)
atexit.register(data_logger.close)

catalog = Catalog(app).load(app.config["TRACKS_CATALOG"])
catalog.upload_tracks(tracks_redis.connection)
catalog.upload_artists(artists_redis.connection)

catalog.upload_recommendations(
    recommendations_lfm_redis.connection,
    "RECOMMENDATIONS_LFM_FILE_PATH",
    key_object="item_id",
    key_recommendations="recommendations",
)
catalog.upload_recommendations(
    recommendations_sasrec_redis.connection,
    "RECOMMENDATIONS_SASREC_FILE_PATH",
    key_object="item_id",
    key_recommendations="recommendations",
)

random_recommender = Random(tracks_redis.connection)

sasrec_i2i_recommender = I2IRecommender(
    listen_history_redis.connection,
    recommendations_sasrec_redis.connection,
    random_recommender,
)
lightfm_i2i_recommender = I2IRecommender(
    listen_history_redis.connection,
    recommendations_lfm_redis.connection,
    random_recommender,
)

ml_artifacts = load_artifacts(app.config["ML_ARTIFACTS_DIR"])
if ml_artifacts is None:
    app.logger.warning(
        "ML artifacts unavailable - treatment will fall back to SasRec-I2I"
    )
    treatment_recommender = sasrec_i2i_recommender
else:
    app.logger.info(
        "Loaded ML artifacts: %d embeddings (dim=%d)",
        ml_artifacts.embeddings.matrix.shape[0],
        ml_artifacts.embeddings.dim,
    )
    treatment_recommender = MLReranker(
        listen_history_redis=listen_history_redis.connection,
        sasrec_redis=recommendations_sasrec_redis.connection,
        lightfm_redis=recommendations_lfm_redis.connection,
        artifacts=ml_artifacts,
        fallback=sasrec_i2i_recommender,
        history_limit=LISTEN_HISTORY_LIMIT,
    )

parser = reqparse.RequestParser()
parser.add_argument("track", type=int, location="json", required=True)
parser.add_argument("time", type=float, location="json", required=True)


def persist_user_listen_history(user: int, track: int, track_time: float) -> None:
    user_history_key = f"user:{user}:listens"
    history_entry = json.dumps({"track": track, "time": track_time})
    listen_history_redis.connection.lpush(user_history_key, history_entry)
    listen_history_redis.connection.ltrim(
        user_history_key, 0, LISTEN_HISTORY_LIMIT - 1
    )


class Hello(Resource):
    def get(self):
        return {
            "status": "alive",
            "message": "welcome to botify, the best toy music recommender",
        }


class Track(Resource):
    def get(self, track: int):
        data = tracks_redis.connection.get(track)
        if data is None:
            abort(404, description="Track not found")
        return asdict(catalog.from_bytes(data))


def _safe_recommend(recommender, user: int, prev_track: int, prev_time: float) -> int:
    """Wrap a recommender call so the HTTP handler is never poisoned by a
    runtime error or a ``None`` recommendation. The previously played track is
    a guaranteed-valid action because the simulator just sent it to us."""
    try:
        recommendation = recommender.recommend_next(user, prev_track, prev_time)
    except Exception:  # pylint: disable=broad-except
        app.logger.exception("Recommender failed for user=%s; using fallback", user)
        return int(prev_track)
    if recommendation is None:
        app.logger.warning("Recommender returned None for user=%s; using fallback", user)
        return int(prev_track)
    return int(recommendation)


class NextTrack(Resource):
    def post(self, user: int):
        start = time.time()

        args = parser.parse_args()
        persist_user_listen_history(user, args.track, args.time)

        treatment = Experiments.HW2.assign(user)
        if treatment == Treatment.C:
            recommender = sasrec_i2i_recommender
        else:
            recommender = treatment_recommender

        recommendation = _safe_recommend(recommender, user, args.track, args.time)

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
