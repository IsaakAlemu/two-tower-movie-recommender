import numpy as np
import torch

from data.preprocess import fit_train, train, test, user_to_idx, movie_to_idx
from models.metrics import to_mask, rank_metrics, relevant_mask_filtered


def main():
    np.random.seed(42)

    n_users = len(user_to_idx)
    n_movies = len(movie_to_idx)
    k = 10

    U = np.random.normal(0, 0.1, size=(n_users, k))
    V = np.random.normal(0, 0.1, size=(n_movies, k))
    global_mean = fit_train["rating"].mean()
    user_bias = np.zeros(n_users)
    movie_bias = np.zeros(n_movies)

    def predict(u, m):
        return global_mean + user_bias[u] + movie_bias[m] + np.dot(U[u], V[m])

    # Same vocabulary as the two-tower model (built from fit_train only),
    # trained on fit_train ratings only -- so the two models see exactly
    # the same interactions and exactly the same catalog.
    rows = fit_train.dropna(subset=["user_idx", "movie_idx"])
    u_idx = rows["user_idx"].to_numpy(dtype=int)
    m_idx = rows["movie_idx"].to_numpy(dtype=int)
    r_arr = rows["rating"].to_numpy(dtype=float)
    n = len(rows)

    lr, reg, epochs = 0.01, 0.02, 20

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        for i in perm:
            u, m, r = u_idx[i], m_idx[i], r_arr[i]
            err = r - predict(u, m)
            uu, vv = U[u].copy(), V[m].copy()
            U[u] += lr * (err * vv - reg * uu)
            V[m] += lr * (err * uu - reg * vv)
            user_bias[u] += lr * (err - reg * user_bias[u])
            movie_bias[m] += lr * (err - reg * movie_bias[m])

        preds = global_mean + user_bias[u_idx] + movie_bias[m_idx] + (U[u_idx] * V[m_idx]).sum(1)
        mse = float(((r_arr - preds) ** 2).mean())
        print(f"Epoch {epoch + 1}/{epochs} - Training MSE: {mse:.4f}")

    # Full [n_users, n_movies] score matrix, same shape/order as the
    # two-tower's score_matrix output.
    scores = torch.tensor(
        global_mean + user_bias[:, None] + movie_bias[None, :] + U @ V.T,
        dtype=torch.float32,
    )

    test_seen = to_mask(train, n_users, n_movies)
    test_relevant = relevant_mask_filtered(test, movie_to_idx, n_users, n_movies, user_to_idx)

    print("\nMF (fair -- fit_train vocab, same harness):",
          rank_metrics(scores, test_seen, test_relevant))


if __name__ == "__main__":
    main()