from dataclasses import dataclass
from typing import List

import marshmallow_dataclass


@dataclass
class ArtistsConfig:
    model: str
    years: List[int]
    top_by_years: int
    top_by_genre: int
    top_by_country: int
    artists_path: str


@dataclass
class TrackCatalogConfig:
    tracks_data_model: str
    tracks_per_artist: int
    tracks_raw_path: str
    tracks_path: str
    tracks_embeddings_model: str
    tracks_embeddings_path: str


@dataclass
class UserCatalogConfig:
    model: str
    users: int
    user_catalog_path: str
    default_interest_neighbours: int = 10
    default_consume_bias: float = 5.0
    default_consume_sharpness: float = 1.0
    default_session_budget: int = 5
    default_artist_discount_gamma: float = 0.8


@dataclass()
class RemoteRecommenderConfig:
    host: str
    port: int


@dataclass
class RecEnvConfig:
    artists_config: ArtistsConfig
    track_catalog_config: TrackCatalogConfig
    user_catalog_config: UserCatalogConfig
    remote_recommender_config: RemoteRecommenderConfig


RecEnvConfigSchema = marshmallow_dataclass.class_schema(RecEnvConfig)
