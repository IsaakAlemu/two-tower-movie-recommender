import torch

from data.preprocess import (
    train,              # fit_train + val combined -- the full "seen" set at test time
    test,
    positive_fit_train,
    user_features,
    movie_features,
    movie_to_idx,
    user_to_idx,
)
from models.two_tower import UserTower, MovieTower
from models.metrics import (
    to_mask,
    rank_metrics,
    relevant_mask_filtered,
    popularity_scores,
    random_scores,
    score_matrix,
)


def main():
    n_users = len(user_to_idx)
    n_movies = len(movie_to_idx)

    user_features_t = torch.tensor(user_features.to_numpy(), dtype=torch.float32)
    movie_features_t = torch.tensor(movie_features.to_numpy(), dtype=torch.float32)

    # Seen mask at test time: everything in `train` (all ratings, not just
    # positives) -- same definition mf.py uses.
    test_seen = to_mask(train, n_users, n_movies)

    # Relevant mask: rating>=4 in test AND movie known to the catalog.
    # This is the fix for the mismatch found in the previous review --
    # mf.py already filters this way; the old evaluate.py didn't, which
    # unfairly penalized the two-tower model on ~87 unreachable interactions.
    test_relevant = relevant_mask_filtered(test, movie_to_idx, n_users, n_movies, user_to_idx)

    # -------------------------------------------------------------
    # Baselines, through the exact same harness
    # -------------------------------------------------------------
    print("Random baseline:", rank_metrics(
        random_scores(n_users, n_movies), test_seen, test_relevant))

    print("Popularity baseline:", rank_metrics(
        popularity_scores(positive_fit_train, n_users, n_movies), test_seen, test_relevant))

    # -------------------------------------------------------------
    # Two-tower model
    # -------------------------------------------------------------
    user_tower = UserTower(n_users, feat_dim=user_features_t.size(1))
    movie_tower = MovieTower(n_movies, feat_dim=movie_features_t.size(1))

    checkpoint = torch.load("artifacts/two_tower.pt", weights_only=True)
    user_tower.load_state_dict(checkpoint["user_tower"])
    movie_tower.load_state_dict(checkpoint["movie_tower"])

    scores = score_matrix(user_tower, movie_tower, user_features_t, movie_features_t)

    # Sanity check: fit quality on TRAIN positives, no seen-filtering.
    # This should score high -- if it doesn't, something upstream is
    # still broken (this isn't a generalization number, it's "did it learn
    # the training data at all").
    train_pos_mask = to_mask(train[train["rating"] >= 4], n_users, n_movies)
    no_filter = torch.zeros_like(test_seen)
    print("Train-fit sanity check:", rank_metrics(scores, no_filter, train_pos_mask))

    print("Two-tower test:", rank_metrics(scores, test_seen, test_relevant))


if __name__ == "__main__":
    main()