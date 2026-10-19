"""
Build a FAISS retrieval index from the trained two-tower checkpoint,
save serving metadata, and provide a simple recommend(user_id, k)
query function.

pip install faiss-cpu
"""

import faiss
import json
import numpy as np
import torch

from data.preprocess import (
    train,
    user_features,
    movie_features,
    movie_to_idx,
    user_to_idx,
    movies,
    genre_names,
)
from models.two_tower import UserTower, MovieTower


IDX_TO_MOVIE = {
    idx: movie_id
    for movie_id, idx in movie_to_idx.items()
}


def build_serving_metadata():
    """Save everything the API needs without requiring raw MovieLens data."""

    movie_metadata = {
        int(row["movie_id"]): {
            "title": row["title"],
            "genres": [
                genre
                for genre in genre_names
                if row[genre] == 1
            ],
        }
        for _, row in movies.iterrows()
        if row["movie_id"] in movie_to_idx
    }

    seen_movies = {
        int(user_id): [
            int(movie_id)
            for movie_id in train.loc[
                train["user_id"] == user_id,
                "movie_id"
            ]
            if movie_id in movie_to_idx
        ]
        for user_id in user_to_idx
    }

    metadata = {
        "user_to_idx": {
            str(user_id): int(idx)
            for user_id, idx in user_to_idx.items()
        },

        # FAISS returns an integer index. This lets the API
        # convert that index back into the original MovieLens ID.
        "idx_to_movie": {
            str(idx): int(movie_id)
            for idx, movie_id in IDX_TO_MOVIE.items()
        },

        "movie_metadata": movie_metadata,
        "seen_movies": seen_movies,
    }

    with open(
        "artifacts/serving_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f)

    print("Saved serving metadata.")


def build_index():
    n_users = len(user_to_idx)
    n_movies = len(movie_to_idx)

    user_features_t = torch.tensor(
        user_features.to_numpy(),
        dtype=torch.float32
    )

    movie_features_t = torch.tensor(
        movie_features.to_numpy(),
        dtype=torch.float32
    )

    user_tower = UserTower(
        n_users,
        feat_dim=user_features_t.size(1)
    )

    movie_tower = MovieTower(
        n_movies,
        feat_dim=movie_features_t.size(1)
    )

    checkpoint = torch.load(
        "artifacts/two_tower.pt",
        weights_only=True
    )

    user_tower.load_state_dict(
        checkpoint["user_tower"]
    )

    movie_tower.load_state_dict(
        checkpoint["movie_tower"]
    )

    user_tower.eval()
    movie_tower.eval()

    with torch.no_grad():
        movie_vectors = movie_tower(
            torch.arange(n_movies),
            movie_features_t
        )

        user_vectors = user_tower(
            torch.arange(n_users),
            user_features_t
        )

    movie_vectors_np = (
        movie_vectors
        .numpy()
        .astype("float32")
    )

    user_vectors_np = (
        user_vectors
        .numpy()
        .astype("float32")
    )

    # Vectors are already L2-normalized, so inner product == cosine
    # similarity. IndexFlatIP is exact and fast for 1,582 movies.
    index = faiss.IndexFlatIP(
        movie_vectors_np.shape[1]
    )

    index.add(movie_vectors_np)

    faiss.write_index(
        index,
        "artifacts/movie_index.faiss"
    )

    np.save(
        "artifacts/user_vectors.npy",
        user_vectors_np
    )

    build_serving_metadata()

    print(
        f"Indexed {index.ntotal} movies, "
        f"{n_users} user vectors cached."
    )

    return index, user_vectors_np


def _seen_movie_idx(user_id):
    seen_ids = train.loc[
        train["user_id"] == user_id,
        "movie_id"
    ]

    return {
        movie_to_idx[m]
        for m in seen_ids
        if m in movie_to_idx
    }


def recommend(
    user_id,
    k=10,
    index=None,
    user_vectors_np=None
):
    """Top-k movie_ids for user_id, with already-seen movies filtered out."""

    if index is None:
        index = faiss.read_index(
            "artifacts/movie_index.faiss"
        )

    if user_vectors_np is None:
        user_vectors_np = np.load(
            "artifacts/user_vectors.npy"
        )

    if user_id not in user_to_idx:
        raise ValueError(
            f"user_id {user_id} not in fit_train vocabulary "
            f"-- needs a cold-start path, not this index."
        )

    user_idx = user_to_idx[user_id]

    query = user_vectors_np[
        user_idx:user_idx + 1
    ]

    seen = _seen_movie_idx(user_id)

    # Over-fetch to survive filtering out seen items,
    # then trim to k.
    fetch_k = min(
        index.ntotal,
        k + len(seen) + 10
    )

    scores, indices = index.search(
        query,
        fetch_k
    )

    recs = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):
        if idx == -1 or idx in seen:
            continue

        recs.append(
            (
                IDX_TO_MOVIE[int(idx)],
                float(score)
            )
        )

        if len(recs) == k:
            break

    return recs


if __name__ == "__main__":
    index, user_vectors_np = build_index()

    example_user = next(
        iter(user_to_idx)
    )

    print(
        f"\nTop-10 for user {example_user}:"
    )

    for movie_id, score in recommend(
        example_user,
        k=10,
        index=index,
        user_vectors_np=user_vectors_np
    ):
        print(
            f"  movie_id={movie_id}  "
            f"score={score:.4f}"
        )