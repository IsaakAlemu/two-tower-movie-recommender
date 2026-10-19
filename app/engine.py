"""
Serving-side recommendation engine.

The API uses only pre-built artifacts:
- FAISS movie index
- cached user vectors
- serving metadata

It does not load the raw MovieLens dataset or training code.
"""

import json

import faiss
import numpy as np


_index = None
_user_vectors = None

_user_to_idx = None
_idx_to_movie = None
_movie_metadata = None
_seen_movies = None


def load_metadata():
    """Load serving metadata from the pre-built JSON artifact."""

    global _user_to_idx, _idx_to_movie, _movie_metadata, _seen_movies

    if _user_to_idx is None:
        with open(
            "artifacts/serving_metadata.json",
            "r",
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)

        _user_to_idx = {
            int(user_id): int(idx)
            for user_id, idx in metadata["user_to_idx"].items()
        }

        _idx_to_movie = {
            int(idx): int(movie_id)
            for idx, movie_id
            in metadata["idx_to_movie"].items()
        }

        _movie_metadata = {
            int(movie_id): movie_data
            for movie_id, movie_data
            in metadata["movie_metadata"].items()
        }

        _seen_movies = {
            int(user_id): set(movie_ids)
            for user_id, movie_ids
            in metadata["seen_movies"].items()
        }

    return (
        _user_to_idx,
        _idx_to_movie,
        _movie_metadata,
        _seen_movies,
    )


def load_artifacts():
    """Load the FAISS index and cached user vectors once."""

    global _index, _user_vectors

    if _index is None:
        _index = faiss.read_index(
            "artifacts/movie_index.faiss"
        )

        _user_vectors = np.load(
            "artifacts/user_vectors.npy"
        )

    return _index, _user_vectors


def get_recommendations(user_id: int, k: int = 10):
    """Return top-k unseen movie recommendations for a known user."""

    (
        user_to_idx,
        idx_to_movie,
        movie_metadata,
        seen_movies,
    ) = load_metadata()

    if user_id not in user_to_idx:
        raise ValueError(
            f"user_id {user_id} is not in the trained vocabulary "
            f"(cold-start users aren't supported by this index yet)."
        )

    index, user_vectors = load_artifacts()

    user_idx = user_to_idx[user_id]

    query = user_vectors[
        user_idx:user_idx + 1
    ]

    seen = seen_movies.get(
        user_id,
        set()
    )

    fetch_k = min(
        index.ntotal,
        k + len(seen) + 10
    )

    scores, indices = index.search(
        query,
        fetch_k
    )

    recommendations = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):
        if index_id == -1:
            continue

        if index_id not in idx_to_movie:
            continue

        movie_id = idx_to_movie[int(index_id)]

        if movie_id in seen:
            continue

        metadata = movie_metadata.get(
            movie_id,
            {}
        )

        recommendations.append(
            {
                "movie_id": movie_id,
                "title": metadata.get(
                    "title",
                    "Unknown",
                ),
                "score": float(score),
                "genres": metadata.get(
                    "genres",
                    [],
                ),
            }
        )

        if len(recommendations) == k:
            break

    return recommendations