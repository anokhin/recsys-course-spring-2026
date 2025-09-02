import json
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request

import streamlit as st
from streamlit_autorefresh import st_autorefresh

TIMEOUT_SECONDS = 300
MAX_LOG = 2000


@dataclass
class Pending:
    id: str
    received_at: str
    path: str
    user: int
    json_body: Any
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[Dict[str, Any]] = None


class RequestStore:
    def __init__(self, maxlen: int = MAX_LOG):
        self._log: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._pending: Dict[str, Pending] = {}
        self._lock = threading.Lock()

    def add_log(self, item: Dict[str, Any]) -> None:
        with self._lock:
            self._log.appendleft(item)

    def logs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._log)

    def clear_logs(self) -> None:
        with self._lock:
            self._log.clear()

    def put_pending(self, p: Pending) -> None:
        with self._lock:
            self._pending[p.id] = p

    def get_pending(self, pid: str) -> Optional[Pending]:
        with self._lock:
            return self._pending.get(pid)

    def pop_pending(self, pid: str) -> Optional[Pending]:
        with self._lock:
            return self._pending.pop(pid, None)

    def list_pending(self) -> List[Pending]:
        with self._lock:
            return sorted(
                self._pending.values(), key=lambda x: x.received_at, reverse=True
            )


@st.cache_resource(show_spinner=False)
def get_store():
    print("Creating store")
    return RequestStore()


store = get_store()

flask_app = Flask(__name__)


@flask_app.route("/info", methods=["GET"])  # simple health check
def info():
    return jsonify({"ok": True, "service": "streamlit botify demo"}), 200


@flask_app.route("/next/<int:user>", methods=["POST"])
def recommend(user):
    ts = datetime.now().strftime("%H:%M:%S")
    pid = uuid.uuid4().hex[:5]

    payload = request.get_json(force=True, silent=False)

    pending = Pending(
        id=pid,
        received_at=ts,
        path=request.path,
        json_body=payload,
        user=user,
    )

    # Add to pending & log
    store.put_pending(pending)
    store.add_log(
        {
            "id": pid,
            "received_at": ts,
            "path": pending.path,
            "user": user,
            "json": payload,
            "status": "pending",
        }
    )

    responded = pending.event.wait(timeout=TIMEOUT_SECONDS)

    store.pop_pending(pid)

    if not responded or pending.response is None:
        # Timed out
        msg = {
            "ok": False,
            "status": "timeout",
            "message": "No manual response within timeout",
            "request_id": pid,
        }
        store.add_log(
            {
                "id": pid,
                "received_at": datetime.now().strftime("%H:%M:%S"),
                "status": "timeout",
                "user": user,
            }
        )
        return jsonify(msg), 202
    else:
        store.add_log(
            {
                "id": pid,
                "received_at": datetime.now().strftime("%H:%M:%S"),
                "status": "responded",
                "response": pending.response,
                "user": user,
            }
        )
        return jsonify(pending.response), 200


@flask_app.route("/last/<int:user>", methods=["POST"])
def last(user):
    return jsonify({"user": user}), 200


_FLASK_STARTED = False


@st.cache_resource(show_spinner=False)
def load_tracks():
    tracks = {}
    with open("data/tracks.json") as tracks_file:
        print("Loading tracks")
        for line in tracks_file:
            track_record = json.loads(line)
            tracks[track_record["track"]] = track_record
    return tracks


def time_reaction(time):
    return "😄" if time > 0.8 else ("😢" if time < 0.2 else "😐")


@st.cache_resource(show_spinner=False)
def start_flask_server(host: str = "0.0.0.0", port: int = 5001) -> Tuple[str, int]:
    global _FLASK_STARTED
    if not _FLASK_STARTED:

        def run():
            flask_app.run(
                host=host, port=port, debug=False, use_reloader=False, threaded=True
            )

        t = threading.Thread(target=run, name="FlaskServer", daemon=True)
        t.start()
        _FLASK_STARTED = True
    return host, port


def draw_sidebar(pendings):
    st.sidebar.header("Pending requests")

    if not pendings:
        st.sidebar.info("No pending requests")
    else:
        for pending in pendings:
            st.sidebar.caption(
                f"User: {pending.user} Request: {pending.id}@{pending.received_at}"
            )

    st.sidebar.subheader("Auto refresh")
    enable_auto = st.sidebar.toggle("Enable", value=True)
    interval_ms = st.sidebar.slider("Interval (ms)", 500, 10000, 1000, 500)

    if enable_auto:
        st_autorefresh(interval=interval_ms, key="auto_refresh")


def draw_requests(selected, tracks):
    # Show incoming JSON
    request_col, response_col = st.columns([1, 1])
    with request_col:
        st.subheader(f"Current request from user {selected.user}")

        track = tracks[selected.json_body["track"]]
        st.text(
            f"Previous track: {track['title']} by {track['artist']} ({', '.join(track['genres'])})"
        )
        time = selected.json_body["time"]
        st.text(f"Previous track time: {time} {time_reaction(time)}")

    with response_col:
        st.subheader("What do you recommend?")
        recommendation = st.selectbox(
            "Track",
            options=list(tracks.keys()),
            format_func=lambda t: f"{tracks[t]['title']} by '{tracks[t]['artist']}' ({tracks[t]['artist_genre']})",
            index=0,
        )

        send_col, cancel_col = st.columns([1, 1])
        with send_col:
            if st.button("✅ Send Response", use_container_width=True):
                try:
                    parsed = {"user": selected.user, "track": recommendation}
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
                else:
                    target = store.get_pending(selected.id)
                    if target is None:
                        st.warning(
                            "This request is no longer pending (likely timed out or already approved)."
                        )
                    else:
                        target.response = parsed
                        target.event.set()
                        st.toast("Response sent ✅")

        with cancel_col:
            if st.button("🛑 Cancel / Reject (send 202)", use_container_width=True):
                target = store.get_pending(selected.id)
                if target is None:
                    st.warning("This request is no longer pending.")
                else:
                    target.event.set()  # no approved_response => Flask returns 202
                    st.toast("Marked as rejected / timed out")


def draw_logs(entries, tracks):
    st.subheader("User history")

    rows = []
    for i, entry in enumerate(entries):
        request_data = entry["json"]
        track = tracks[request_data["track"]]
        time = request_data["time"]
        rows.append(
            {
                "received_at": entry.get("received_at"),
                "path": entry.get("path", ""),
                "track": track["title"],
                "artist": track["artist"],
                "genres": ", ".join(track["genres"]),
                "time": time,
                "reaction": time_reaction(time),
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)

    if st.button("🧹 Clear log", type="secondary", use_container_width=False):
        store.clear_logs()
        st.toast("Cleared request log")


def draw_main_screen(pendings, tracks):
    if pendings:
        selected_id = st.selectbox(
            "Select a pending request",
            options=[p.id for p in pendings],
            format_func=lambda x: f"{x} from user {next(p for p in pendings if p.id == x).user}",
            index=0,
        )
        selected = next(p for p in pendings if p.id == selected_id)

        draw_requests(selected, tracks)

        st.divider()

        entries = [
            entry
            for entry in store.logs()
            if entry["user"] == selected.user and entry["status"] == "pending"
        ]
        if entries:
            draw_logs(entries, tracks)
        else:
            st.info("No log entries yet")
    else:
        st.info("No pending requests")

    st.divider()
    host, port = start_flask_server()
    st.success(f"Flask server running at http://{host}:{port}")


def demo():
    st.set_page_config(page_title="Botify demo", layout="wide")
    st.title("💿 Botify demo")

    pendings = get_store().list_pending()
    tracks = load_tracks()

    draw_sidebar(pendings)
    draw_main_screen(pendings, tracks)


demo()
