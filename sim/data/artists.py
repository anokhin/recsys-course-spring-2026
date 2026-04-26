import json
import logging
import sys

import tqdm
import yaml

from ollama import ChatResponse
from ollama import chat

from utils import *

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

GENRES = [
    "Pop",
    "Hip-Hop / Rap",
    "Rock",
    "R&B",
    "Electronic / Dance (EDM)",
    "Latin",
    "K-Pop",
    "Country",
    "Jazz",
    "Classical",
    "Reggae",
    "Metal",
    "Blues",
    "Soul",
    "Funk",
    "Disco",
    "Indie / Alternative",
    "Folk",
    "Gospel / Christian",
    "Afrobeats",
]

COUNTRIES = [
    "United States",
    "United Kingdom",
    "Canada",
    "Australia",
    "Germany",
    "Sweden",
    "Japan",
    "Brazil",
    "Ireland",
    "South Korea",
    "Russia",
]


def get_country_artists(model, country: str, n=20):
    response: ChatResponse = chat(
        model,
        messages=[
            {
                "role": "user",
                "content": unindent(
                    f"""
            Name top-{n} most popular music artists from {country}. 
            Provide a plain numbered list with artist names in english. 
            Please don't give any additional explanations, just the plain list.
            """
                ),
            },
        ],
    )
    return parse_list_response(response.message.content, country)


def get_genre_artists(model, genre: str, n=20):
    response: ChatResponse = chat(
        model,
        messages=[
            {
                "role": "user",
                "content": unindent(
                    f"""
            Name top-{n} most popular music artists playing music in genre {genre}. 
            Provide a plain numbered list with artist names in english. 
            Please don't give any additional explanations, just the plain list.
            """
                ),
            },
        ],
    )
    return parse_list_response(response.message.content, genre)


def get_top_artists(model, start_year: int, end_year: int, n: int = 50):
    response: ChatResponse = chat(
        model,
        messages=[
            {
                "role": "user",
                "content": unindent(
                    f"""
            Name top-{n} most popular music artists in the world between {start_year} and {end_year}. 
            Provide a plain numbered list with artists names in english.
            Please don't give any additional explanations, just the plain list.
            """
                ),
            },
        ],
    )
    return parse_list_response(response.message.content, "top")


def normalize_artist(model, artist):
    response: ChatResponse = chat(
        model,
        messages=[
            {
                "role": "user",
                "content": unindent(
                    f"""
            Here is a non-normalized music artist name: '{artist}'. 
            Please normalize it, i.e. retain only the official name, stripping all the featuring artists, collaborators and other additional info.
            The result should be a plain string without any parentheses, just the artist name. 
            Do not provide any additional text or explanations.
            """
                ),
            },
        ],
    )
    return response.message.content


def get_artist_data(model, artist: str, genres):
    response: ChatResponse = chat(
        model,
        messages=[
            {
                "role": "user",
                "content": unindent(
                    f"""
            Provide extended information about the music artist '{artist}'. 
            The output should be be a valid json object with the following fields:
            artist (artist name),
            country (artist country of origin), 
            genre (the genre from the list [{', '.join(genres)}] that describes the artist's music best. If no genre fits, use 'unknown')
            genres (a list of main genres of this artist's songs), 
            fans (an estimated number of fans in the world as a number in millions),
            Please only return the json response, no other data or explanations is required. 
            Also remove any formatting.
            """
                ),
            },
        ],
    )

    artist_data = json.loads(response.message.content.replace("\n", ""))

    if artist_data["artist"] != artist:
        logger.error(
            f"artist mismatch. Expected '{artist}', but was '{artist_data['artist']}'"
        )
        artist_data["artist"] = artist

    return artist_data


def get_artists(env_config):
    config = env_config["artists_config"]
    model = config["model"]

    artists = {}

    years = config["years"]
    top_by_years = config["top_by_years"]
    for start_year, end_year in zip(years[:-1], years[1:]):
        logger.info(
            f"Getting top {top_by_years} artists between {start_year} and {end_year}..."
        )
        artists.update(
            {
                normalize_artist(model, record["item"]): f"{start_year}-{end_year}"
                for record in get_top_artists(model, start_year, end_year, top_by_years)
            }
        )

    top_by_genre = config["top_by_genre"]
    for genre in GENRES:
        logger.info(f"Getting top {top_by_genre} '{genre}' artists...")
        genre_artists = get_genre_artists(model, genre, top_by_genre)
        artists.update(
            {normalize_artist(model, record["item"]): genre for record in genre_artists}
        )

    top_by_country = config["top_by_country"]
    for country in COUNTRIES:
        logger.info(f"Getting top {top_by_country} artists from {country}...")
        country_artists = get_country_artists(model, country, top_by_country)
        artists.update(
            {
                normalize_artist(model, record["item"]): country
                for record in country_artists
            }
        )

    logger.info("Getting artists data...")
    artist_records = []
    for index, artist in tqdm.tqdm(enumerate(artists.keys()), total=len(artists)):
        try:
            record = retry(lambda: get_artist_data(model, artist, GENRES))
            record["artist_id"] = index
            artist_records.append(record)
        except ValueError:
            logger.error(f"Error while getting data for {artist}, ")

    with open(config["artists_path"], "w") as artists_file:
        json.dump(artist_records, artists_file)

    logger.info("Saved artist data to: " + config["artists_path"])


def main():
    with open("config/env.yml") as config_file:
        env_config = yaml.safe_load(config_file)
    get_artists(env_config)


if __name__ == "__main__":
    main()
