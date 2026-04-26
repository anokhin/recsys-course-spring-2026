import json
import logging
import os
import sys
from collections import defaultdict

import yaml
import tqdm
import functools

from ollama import chat
from ollama import ChatResponse

import pandas as pd

from utils import *

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

MOODS = [
    "Happy",
    "Sad",
    "Energetic",
    "Calm",
    "Romantic",
    "Melancholic",
    "Angry",
    "Chill",
    "Motivational",
    "Uplifting",
    "Dark",
    "Dreamy",
    "Nostalgic",
    "Playful",
    "Mellow",
    "Epic",
    "Sentimental",
    "Hopeful",
    "Lonely",
    "Intense",
]


def get_tracks(model, artist: str, n: int = 10):
    response: ChatResponse = chat(
        model,
        messages=[
            {
                "role": "user",
                "content": unindent(
                    f"""
            List {n} popular tracks of the music artist '{artist}'. 
            Provide a plain numbered list containing only the track name in english. 
            Please don't give any additional explanations.
            """
                ),
            },
        ],
    )
    return parse_list_response(response.message.content)


def get_track_data(model, artist: str, track: str):
    response: ChatResponse = chat(
        model,
        messages=[
            {
                "role": "user",
                "content": unindent(
                    f"""
            Provide extended information about the track '{track}' by '{artist}'. 
            The output should be be a valid json object with the following REQUIRED fields:
            title (the title of the track), 
            artist (the name of the artist),
            genres (genres of the track as a list of strings), 
            year (the year of the first release of the track), 
            summary (the short track, including lyrics summary or other relevant information, maximum 2000 characters),
            mood (the mood that characterises this song. select exactly one option from the list: {','.join(MOODS)})
            Please only return the json response, no other data or explanations is required.
            Also remove any formatting.
            """
                ),
            },
        ],
    )
    track_data = json.loads(response.message.content.replace("\n", ""))

    alternative_artist = track_data.get("artist", "none")
    if alternative_artist != artist:
        logger.debug(
            f"Track artist mismatch. Expected '{artist}', but was '{alternative_artist}'."
        )
        track_data["alternative_artist"] = alternative_artist
        track_data["artist"] = artist

    alternative_title = track_data.get("title", "none")
    if alternative_title != track:
        logger.debug(
            f"Track title mismatch. Expected '{track}', but was '{alternative_title}'."
        )
        track_data["alternative_title"] = alternative_title
        track_data["title"] = track

    return track_data


def load_saved_tracks(tracks_raw_path):
    loaded_tracks = defaultdict(set)

    if not os.path.exists(tracks_raw_path):
        logger.info("Previously loaded tracks not found")
        return loaded_tracks

    with open(tracks_raw_path) as tracks_file:
        for line in tracks_file:
            if not line:
                continue
            record = json.loads(line)
            loaded_tracks[record["artist"]].add(record["title"])

    logger.info(f"Previously loaded tracks found for {len(loaded_tracks)} artists")
    return loaded_tracks


def save_raw_tracks(env_config):
    with open(env_config["artists_config"]["artists_path"], "r") as artists_file:
        artist_records = json.load(artists_file)

    config = env_config["track_catalog_config"]

    tracks_raw_path = config["tracks_raw_path"]
    loaded_tracks = load_saved_tracks(tracks_raw_path)

    model = config["tracks_data_model"]
    tracks_per_artist = config["tracks_per_artist"]

    with open(tracks_raw_path, "a", buffering=1) as tracks_file:
        for artist_record in tqdm.tqdm(artist_records):
            artist = artist_record["artist"]

            loaded_artist_tracks = loaded_tracks[artist]
            if len(loaded_artist_tracks) >= tracks_per_artist:
                continue

            tracks = retry(lambda: get_tracks(model, artist, tracks_per_artist))

            for track_data in list(tracks):
                track = track_data["item"]
                if track in loaded_artist_tracks:
                    continue

                try:
                    record = retry(lambda: get_track_data(model, artist, track))

                    record["artist_id"] = artist_record["artist_id"]
                    record["artist_country"] = artist_record["country"]
                    record["artist_genres"] = artist_record["genres"]
                    record["artist_genre"] = artist_record["genre"]
                    record["artist_fans"] = artist_record["fans"]

                    tracks_file.write(json.dumps(record) + "\n")
                except ValueError:
                    print(f"JSONDecodeError: {artist} {track}")


def normalize_fans(fans):
    if fans == "unknown":
        return 1.0

    if isinstance(fans, str):
        fans = fans.split("-")[0]

    return min(max(float(fans), 1.0), 100.0)


def save_cleaned_tracks(env_config):
    config = env_config["track_catalog_config"]
    tracks_raw_path = config["tracks_raw_path"]

    logging.info("Cleaning up tracks data")

    tracks_data = pd.read_json(tracks_raw_path, lines=True)

    columns = [
        "title",
        "alternative_title",
        "artist",
        "alternative_artist",
        "genres",
        "year",
        "mood",
        "summary",
        "artist_id",
        "artist_country",
        "artist_genres",
        "artist_genre",
        "artist_fans",
    ]

    tracks_data = (
        tracks_data.loc[
            functools.reduce(
                lambda x, y: x & y,
                [
                    pd.notna(tracks_data[column])
                    for column in columns
                    if not column.startswith("alternative")
                ],
            ),
            columns,
        ]
        .copy()
        .drop_duplicates(subset=["title", "artist"])
    )

    tracks_data = tracks_data.reset_index().drop("index", axis=1)
    tracks_data["track"] = tracks_data.index.values

    tracks_data["artist_fans"] = tracks_data["artist_fans"].map(normalize_fans)

    tracks_data.to_json(config["tracks_path"], orient="records", lines=True)
    logger.info(f"Saved cleaned tracks to {config['tracks_path']}")


def main():
    with open("config/env.yml") as config_file:
        env_config = yaml.safe_load(config_file)

    save_raw_tracks(env_config)
    save_cleaned_tracks(env_config)


if __name__ == "__main__":
    main()
