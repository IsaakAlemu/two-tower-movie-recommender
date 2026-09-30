"""
Build a FAISS retrieval index from the trained two-tower checkpoint,
and provide a simple recommend(user_id, k) query function.

pip install faiss-cpu
"""
import faiss
import numpy as np
import torch

from data.preprocess import (
    train,             # fit_train + val -- full known interaction history
    user_features,
    movie_features,
    movie_to_idx,
    user_to_idx,
)
from models.two_tower import UserTower, MovieTower


IDX_TO_MOVIE = {idx: movie_id for movie_id, idx in movie_to_idx.items()}


def build_index():
    n_users = len(user_to_idx)
    n_movies = len(movie_to_idx)

    user_features_t = torch.tensor(user_features.to_numpy(), dtype=torch.float32)
    movie_features_t = torch.tensor(movie_features.to_numpy(), dtype=torch.float32)

    user_tower = UserTower(n_users, feat_dim=user_features_t.size(1))
    movie_tower = MovieTower(n_movies, feat_dim=movie_features_t.size(1))

    checkpoint = torch.load("artifacts/two_tower.pt", weights_only=True)
    user_tower.load_state_dict(checkpoint["user_tower"])
    movie_tower.load_state_dict(checkpoint["movie_tower"])
    user_tower.eval()
    movie_tower.eval()

    with torch.no_grad():
        movie_vectors = movie_tower(torch.arange(n_movies), movie_features_t)
        user_vectors = user_tower(torch.arange(n_users), user_features_t)

    movie_vectors_np = movie_vectors.numpy().astype("float32")
    user_vectors_np = user_vectors.numpy().astype("float32")

    # Vectors are already L2-normalized, so inner product == cosine
    # similarity -- IndexFlatIP is exact (no approximation) and plenty
    # fast for 1,582 items. Move to IVF/HNSW only if the catalog grows
    # into the hundreds of thousands+.
    index = faiss.IndexFlatIP(movie_vectors_np.shape[1])
    index.add(movie_vectors_np)

    faiss.write_index(index, "artifacts/movie_index.faiss")
    np.save("artifacts/user_vectors.npy", user_vectors_np)

    print(f"Indexed {index.ntotal} movies, {n_users} user vectors cached.")
    return index, user_vectors_np


def _seen_movie_idx(user_id):
    seen_ids = train.loc[train["user_id"] == user_id, "movie_id"]
    return {movie_to_idx[m] for m in seen_ids if m in movie_to_idx}


def recommend(user_id, k=10, index=None, user_vectors_np=None):
    """Top-k movie_ids for user_id, with already-seen movies filtered out.
    Pass index/user_vectors_np to reuse a loaded index; otherwise loads
    from disk each call (fine for a demo, cache these in a real service)."""
    if index is None:
        index = faiss.read_index("artifacts/movie_index.faiss")
    if user_vectors_np is None:
        user_vectors_np = np.load("artifacts/user_vectors.npy")

    if user_id not in user_to_idx:
        raise ValueError(f"user_id {user_id} not in fit_train vocabulary "
                          f"-- needs a cold-start path, not this index.")

    user_idx = user_to_idx[user_id]
    query = user_vectors_np[user_idx:user_idx + 1]   # [1, 32]

    seen = _seen_movie_idx(user_id)
    # Over-fetch to survive filtering out seen items, then trim to k.
    fetch_k = k + len(seen) + 10
    scores, indices = index.search(query, fetch_k)

    recs = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx in seen:
            continue
        recs.append((IDX_TO_MOVIE[int(idx)], float(score)))
        if len(recs) == k:
            break
    return recs


if __name__ == "__main__":
    index, user_vectors_np = build_index()
    example_user = next(iter(user_to_idx))
    print(f"\nTop-10 for user {example_user}:")
    for movie_id, score in recommend(example_user, k=10, index=index, user_vectors_np=user_vectors_np):
        print(f"  movie_id={movie_id}  score={score:.4f}")