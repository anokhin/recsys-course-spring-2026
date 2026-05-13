from flask import Flask, request, jsonify
from botify.recommenders.session_gate_ranker import SessionGateRanker
import joblib
import logging

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# Загрузим модель при старте
model = joblib.load("/app/session_gate_rf_bundle.joblib")
ranker = SessionGateRanker(model=model)


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        observation = data.get("observation", {})
        user = observation.get("user")
        prev_track = observation.get("prev_track")
        prev_track_time = observation.get("prev_track_time", 0)
        
        action = ranker.recommend_next(user, prev_track, prev_track_time)
        
        logging.info(f"Recommendation: user={user}, action={action}")
        return jsonify({"action": int(action)})
    except Exception as e:
        logging.error(f"Error in recommend: {e}", exc_info=True)
        return jsonify({"action": 0}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", po
