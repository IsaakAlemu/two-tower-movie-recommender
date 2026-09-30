"""
Serving-side wrapper around the existing retrieval code. This file does
NOT reimplement recommendation logic -- it caches the FAISS index and
user vectors once (so a request doesn't re-read them from disk every
call) and delegates the actual retrieval to models.build_index.recommend(),
the same function used offline for the sanity check.
"""
import faiss
import numpy as np


_index = None
_user_vectors = None

_user_to_idx = None
_id_to_title = None
_id_to_genres = None


def load_metadata():
    """Load MovieLens metadata only when recommendations are requested."""
    global _user_to_idx, _id_to_title, _id_to_genres

    if _user_to_idx is None:
        from data.preprocess import movies, genre_names, user_to_idx

        _user_to_idx = user_to_idx

        _id_to_title = dict(
            zip(
                movies["movie_id"],
                movies["title"],
            )
        )

        _id_to_genres = {
            row["movie_id"]: [
                g for g in genre_names
                if row[g] == 1
            ]
            for _, row in movies.iterrows()
        }

    return _user_to_idx, _id_to_title, _id_to_genres


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
    user_to_idx, id_to_title, id_to_genres = load_metadata()

    if user_id not in user_to_idx:
        raise ValueError(
            f"user_id {user_id} is not in the trained vocabulary "
            f"(cold-start users aren't supported by this index yet)."
        )

    from models.build_index import recommend

    index, user_vectors = load_artifacts()

    recs = recommend(
        user_id,
        k=k,
        index=index,
        user_vectors_np=user_vectors,
    )

    return [
        {
            "movie_id": movie_id,
            "title": id_to_title.get(
                movie_id,
                "Unknown",
            ),
            "score": score,
            "genres": id_to_genres.get(
                movie_id,
                [],
            ),
        }
        for movie_id, score in recs
    ]