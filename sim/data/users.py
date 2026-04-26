import json
import logging
import sys

import numpy as np
import pandas as pd
import tqdm
import yaml
from ollama import ChatResponse
from ollama import chat

from utils import *

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


def get_recommended_artist(model, artist, artists_sample):
    response: ChatResponse = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": unindent(
                    f"""
            Act as a top music expert.
            Here is a list of music artists: {', '.join(artists_sample)}. 
            Pick exactly one artist from this list that you think is most relevant to recommend to someone who likes {artist}.
            Please return your response as a json object with the following fields:
            1) artist: please give the exact name of the recommended artist as it is given in the list.
            2) explanation: briefly explain why do you think that this recommendation is relevant (max 200 characters).
            Please return only this json data, no other explanations are expected.
            If you complete this task to the best of your abilities, you will be tipped $50.
            """
                ),
            },
        ],
    )
    return json.loads(response.message.content.replace("\n", ""))


def sample_user(model: str, user: int, tracks_data):
    user_openness = np.random.uniform(0.25, 0.75)

    artists = tracks_data[["artist", "artist_fans"]].drop_duplicates()

    num_interests = 1 + np.random.binomial(5, 0.5)
    current_artist = None
    interests = []
    for i in range(num_interests):
        if current_artist is None:
            interest_track = tracks_data.sample(n=1, weights="artist_fans").iloc[0]
            current_artist = interest_track["artist"]
            interest = interest_track["track"]
        else:
            artists_sample = artists.sample(n=100, weights="artist_fans")[
                "artist"
            ].to_list()
            response = get_recommended_artist(model, current_artist, artists_sample)
            current_artist = next(
                artist
                for artist in artists_sample
                if artist == response["artist"]
            )

            interest = int(
                tracks_data[tracks_data["artist"] == current_artist]
                .sample(1)
                .iloc[0]["track"]
            )

        interests.append(interest)

        if np.random.random() < user_openness:
            current_artist = None

    consume_bias = np.random.uniform(0.1, 0.6)
    consume_sharpness = np.random.uniform(4, 15)

    return {
        "user": user,
        "interests": interests,
        "consume_bias": consume_bias,
        "consume_sharpness": consume_sharpness,
        "openness": user_openness,
    }


def generate_users(env_config):
    tracks_data = pd.read_json(
        env_config["track_catalog_config"]["tracks_path"], lines=True
    )

    config = env_config["user_catalog_config"]
    users = []
    for j in tqdm.trange(config["users"]):
        try:
            user = retry(lambda: sample_user(config["model"], j, tracks_data))
            users.append(user)
        except ValueError as ve:
            logger.debug(f"Failed to sample user with error: {ve}")

    users_data = pd.DataFrame(users)
    users_data.to_json(config["user_catalog_path"], orient="records", lines=True)


def main():
    with open("config/env.yml") as config_file:
        env_config = yaml.safe_load(config_file)

    generate_users(env_config)


if __name__ == "__main__":
    main()
