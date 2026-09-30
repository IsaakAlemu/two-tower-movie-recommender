import copy
import numpy as np
import torch

from data.preprocess import (
    positive_fit_train,
    fit_train,
    val,
    user_features,
    movie_features,
    movie_to_idx,
    user_to_idx,
)
from models.two_tower import UserTower, MovieTower, full_softmax_loss
from models.metrics import to_mask, rank_metrics, relevant_mask_filtered, score_matrix


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(42)

    n_users = len(user_to_idx)
    n_movies = len(movie_to_idx)

    user_features_t = torch.tensor(user_features.to_numpy(), dtype=torch.float32)
    movie_features_t = torch.tensor(movie_features.to_numpy(), dtype=torch.float32)

    users = torch.as_tensor(positive_fit_train["user_idx"].to_numpy(), dtype=torch.long)
    movies = torch.as_tensor(positive_fit_train["movie_idx"].to_numpy(), dtype=torch.long)
    N = users.numel()
    print(f"Training on {N} positive interactions "
          f"({n_users} users, {n_movies} movies)")

    # Validation masks: seen = everything the user rated in fit_train
    # (matches how the MF baseline defines "seen"); relevant = val
    # positives (val is already a subset of train, so every movie in it
    # is guaranteed to be in the known catalog -- no extra filtering
    # needed here, unlike the final test evaluation).
    val_seen = to_mask(fit_train, n_users, n_movies)
    # val can now contain movies that never appeared in fit_train (that's
    # the whole point of the fix) -- filter relevant_mask to the known
    # catalog the same way test's does, or those users get counted as
    # "evaluated" while being structurally unable to score a hit.
    val_relevant = relevant_mask_filtered(val, movie_to_idx, n_users, n_movies, user_to_idx)

    user_tower = UserTower(n_users, feat_dim=user_features_t.size(1))
    movie_tower = MovieTower(n_movies, feat_dim=movie_features_t.size(1))

    optimizer = torch.optim.AdamW(
        list(user_tower.parameters()) + list(movie_tower.parameters()),
        lr=1e-3,
    )

    tau = 0.07
    batch_size = 512
    max_epochs = 100
    patience = 10

    all_movie_ids = torch.arange(n_movies)

    best_ndcg, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(1, max_epochs + 1):
        user_tower.train()
        movie_tower.train()

        perm = torch.randperm(N)
        total_loss, steps = 0.0, 0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            u_idx, m_idx = users[idx], movies[idx]

            u_vec = user_tower(u_idx, user_features_t[u_idx])
            all_m_vec = movie_tower(all_movie_ids, movie_features_t)

            loss = full_softmax_loss(u_vec, all_m_vec, m_idx, tau)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / steps

        scores = score_matrix(user_tower, movie_tower, user_features_t, movie_features_t)
        val_metrics = rank_metrics(scores, val_seen, val_relevant, ks=(10, 20))

        print(
            f"Epoch {epoch:3d} | loss {avg_loss:.4f} | "
            f"val ndcg@10 {val_metrics['ndcg@10']:.4f} "
            f"recall@10 {val_metrics['recall@10']:.4f} "
            f"hit@10 {val_metrics['hit@10']:.4f} "
            f"cov@10 {val_metrics['coverage@10']:.3f}"
        )

        if val_metrics["ndcg@10"] > best_ndcg:
            best_ndcg = val_metrics["ndcg@10"]
            best_state = (
                copy.deepcopy(user_tower.state_dict()),
                copy.deepcopy(movie_tower.state_dict()),
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(best val ndcg@10 = {best_ndcg:.4f})")
                break

    user_tower.load_state_dict(best_state[0])
    movie_tower.load_state_dict(best_state[1])

    torch.save(
        {"user_tower": user_tower.state_dict(), "movie_tower": movie_tower.state_dict()},
        "artifacts/two_tower.pt",
    )
    print(f"Saved best checkpoint (val ndcg@10 = {best_ndcg:.4f}) to artifacts/two_tower.pt")


if __name__ == "__main__":
    main()