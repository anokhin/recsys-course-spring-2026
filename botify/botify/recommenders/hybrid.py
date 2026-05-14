import pickle
from .recommender import Recommender


class HybridRecommender(Recommender):
    def __init__(self, indexed, i2i, fallback, catalog, redis):
        self.indexed = indexed
        self.i2i = i2i
        self.fallback = fallback
        self.catalog = catalog
        self.redis = redis

    def recommend_next(self, user, prev_track, prev_track_time):

        try:
            # =========================
            # SHOWN
            # =========================
            shown_key = f"user:{user}:shown"
            shown_raw = self.redis.smembers(shown_key)
            shown = set(int(x) for x in shown_raw) if shown_raw else set()

            # =========================
            # 1. SASRec (ОСНОВА)
            # =========================
            sasrec_list = []

            if prev_track is not None and prev_track >= 0:
                data = self.i2i.i2i_redis.get(prev_track)
                if data:
                    try:
                        sasrec_list = list(pickle.loads(data))
                    except:
                        sasrec_list = []

            if not sasrec_list:
                return self.fallback.recommend_next(user, prev_track, prev_track_time)

            # фильтруем уже показанные
            candidates = [int(t) for t in sasrec_list[:50] if int(t) not in shown]

            if not candidates:
                track = int(sasrec_list[0])
                self.redis.sadd(shown_key, track)
                return track

            # =========================
            # 2. USER (слабый сигнал)
            # =========================
            user_set = set()

            rec_bytes = self.indexed.recommendations_redis.get(user)
            if rec_bytes:
                try:
                    user_list = list(self.catalog.from_bytes(rec_bytes))
                    user_set = set(int(t) for t in user_list[:100])
                except:
                    user_set = set()

            # =========================
            # 3. РЕРАНК (АККУРАТНЫЙ)
            # =========================
            best_track = None
            best_score = -1e9

            for rank, track in enumerate(candidates):

                # === БАЗА: SASRec (главный сигнал)
                score = (50 - rank) * 5

                # === USER (очень мягко)
                if track in user_set:
                    score += 10

                # === TIME (очень аккуратно)
                if prev_track_time < 0.2:
                    # пользователь скипает → чуть усиливаем user
                    if track in user_set:
                        score += 5

                elif prev_track_time > 0.8:
                    # пользователь доволен → усиливаем SASRec
                    score += (50 - rank) * 2

                # === НОВИЗНА
                if track not in shown:
                    score += 3

                if score > best_score:
                    best_score = score
                    best_track = track

            if best_track is None:
                return self.fallback.recommend_next(user, prev_track, prev_track_time)

            self.redis.sadd(shown_key, best_track)

            return int(best_track)

        except Exception as e:
            print("HYBRID ERROR:", e)
            return self.fallback.recommend_next(user, prev_track, prev_track_time)