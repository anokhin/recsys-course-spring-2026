import json
import time
import redis
from datetime import datetime

from flask import Flask
from flask_restful import Api, Resource, reqparse

from botify.data import DataLogger, Datum
from botify.experiment import Experiments, Treatment
from botify.recommenders.i2i import I2IRecommender
from botify.recommenders.random import Random
from botify.recommenders.ml_ranker import MLRanker
from botify.track import Catalog

app = Flask(__name__)
app.config.from_file("config.json", load=json.load)
api = Api(app)

# 直接用 redis-py，绕开 flask_redis 的配置兼容问题
tracks_redis = redis.Redis.from_url(app.config["REDIS_TRACKS_URL"])
artists_redis = redis.Redis.from_url(app.config["REDIS_ARTIST_URL"])
listen_history_redis = redis.Redis.from_url(app.config["REDIS_LISTEN_HISTORY_URL"])
recommendations_lfm_redis = redis.Redis.from_url(app.config["REDIS_RECOMMENDATIONS_LFM_URL"])
recommendations_sasrec_redis = redis.Redis.from_url(app.config["REDIS_RECOMMENDATIONS_SASREC_URL"])
recommendations_hstu_redis = redis.Redis.from_url(app.config["REDIS_RECOMMENDATIONS_HSTU_URL"])

data_logger = DataLogger(app)

# 按 track.py 的真实接口初始化
catalog = Catalog(app)
catalog.load(app.config["TRACKS_CATALOG"])
catalog.upload_tracks(tracks_redis)
catalog.upload_artists(artists_redis)

catalog.upload_recommendations(
    recommendations_lfm_redis,
    "RECOMMENDATIONS_LFM_FILE_PATH",
    key_object="item_id",
    key_recommendations="recommendations",
)
catalog.upload_recommendations(
    recommendations_sasrec_redis,
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

parser = reqparse.RequestParser()
parser.add_argument("track", type=int, location="json", required=True)
parser.add_argument("time", type=float, location="json", required=True)

random_recommender = Random(tracks_redis)

# control = SasRec-I2I
sasrec_i2i_recommender = I2IRecommender(
    listen_history_redis,
    recommendations_sasrec_redis,
    random_recommender,
)

# treatment = 保守 ML recommender
ml_recommender = MLRanker(
    "/app/ml_ranker_bundle.joblib",
    recommendations_sasrec_redis,
    tracks_redis,
    listen_history_redis,
    sasrec_i2i_recommender,
    random_recommender,
    topk=20,
    min_prev_time=0.80,
    abs_threshold=0.78,
    margin=0.10,
)


def persist_user_listen_history(user: int, track: int, listened_time: float):
    # 必须和 i2i.py 里读取的 key 一致
    key = f"user:{user}:listens"
    listen_history_redis.lpush(
        key,
        json.dumps({"track": int(track), "time": float(listened_time)}),
    )
    listen_history_redis.ltrim(key, 0, 100)


class Hello(Resource):
    def get(self):
        return {"status": "ready", "message": "Botify is ready"}


class NextTrack(Resource):
    def post(self, user_id: int):
        start = time.time()
        args = parser.parse_args()

        persist_user_listen_history(user_id, args["track"], args["time"])

        treatment = Experiments.AA.assign(user_id)
        if treatment == Treatment.C:
            recommender = sasrec_i2i_recommender
        else:
            recommender = ml_recommender

        try:
            recommendation = recommender.recommend_next(
                user_id, args["track"], args["time"]
            )

            if recommendation is None:
                app.logger.error(
                    f"Recommendation is None. user={user_id}, "
                    f"track={args['track']}, time={args['time']}, bucket={treatment.name}"
                )
                recommendation = random_recommender.recommend_next(
                    user_id, args["track"], args["time"]
                )

            recommendation = int(recommendation)

        except Exception as e:
            app.logger.exception(
                f"Error in recommend_next. user={user_id}, "
                f"track={args['track']}, time={args['time']}, bucket={treatment.name}, err={e}"
            )
            recommendation = int(
                random_recommender.recommend_next(user_id, args["track"], args["time"])
            )

        data_logger.log(
            "next",
            Datum(
                int(datetime.now().timestamp() * 1000),
                user_id,
                args["track"],
                args["time"],
                time.time() - start,
                recommendation,
            ),
            experiments={"AA": treatment.name},
        )

        return {"user": user_id, "track": recommendation}


class LastTrack(Resource):
    def post(self, user_id: int):
        start = time.time()
        args = parser.parse_args()

        persist_user_listen_history(user_id, args["track"], args["time"])

        treatment = Experiments.AA.assign(user_id)
        data_logger.log(
            "last",
            Datum(
                int(datetime.now().timestamp() * 1000),
                user_id,
                args["track"],
                args["time"],
                time.time() - start,
                None,
            ),
            experiments={"AA": treatment.name},
        )

        return {"user": user_id}


api.add_resource(Hello, "/")
api.add_resource(NextTrack, "/next/<int:user_id>")
api.add_resource(LastTrack, "/last/<int:user_id>")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
