import logging
import sys

import numpy as np
import pandas as pd
import tqdm
import yaml

from utils import *

from ollama import embed

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


def get_track_embedding(model, track_record):
    prompt = unindent(
        f"""
        This is the description of track '{track_record['title']}' released in {track_record['year']} 
        by '{track_record['artist']}' from '{track_record['artist_country']}'.
        The the artists genres are {', '.join(track_record['artist_genres'])}.
        The track genres are {', '.join(track_record['genres'])} and its mood is {track_record.get('mood', 'unknown')}.
        Track info and lyrics summary: {track_record['summary']}
        """
    ).strip()
    return embed(model, input=prompt)["embeddings"][0]


def save_track_embeddings(env_config):
    logger.info("Getting track embeddings")
    config = env_config["track_catalog_config"]
    model = config["tracks_embeddings_model"]

    tracks_data = pd.read_json(config["tracks_path"], lines=True)
    embeddings = []
    for index, row in tqdm.tqdm(tracks_data.iterrows(), total=len(tracks_data)):
        embeddings.append(get_track_embedding(model, row))
    np.save(config["tracks_embeddings_path"], embeddings)


def main():
    with open("config/env.yml") as config_file:
        env_config = yaml.safe_load(config_file)

    save_track_embeddings(env_config)


if __name__ == "__main__":
    main()
