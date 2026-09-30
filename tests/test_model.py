import torch

from models.two_tower import UserTower, MovieTower


def test_user_tower_output_shape():
    model = UserTower(
        num_users=10,
        feat_dim=27,
    )

    user_ids = torch.tensor([0, 1, 2, 3])
    user_features = torch.randn(4, 27)

    output = model(user_ids, user_features)

    assert output.shape == (4, 32)


def test_movie_tower_output_shape():
    model = MovieTower(
        num_movies=20,
        feat_dim=20,
    )

    movie_ids = torch.tensor([0, 1, 2, 3])
    movie_features = torch.randn(4, 20)

    output = model(movie_ids, movie_features)

    assert output.shape == (4, 32)


def test_tower_outputs_are_normalized():
    model = UserTower(
        num_users=10,
        feat_dim=27,
    )

    user_ids = torch.tensor([0, 1, 2, 3])
    user_features = torch.randn(4, 27)

    output = model(user_ids, user_features)

    norms = torch.linalg.vector_norm(output, dim=1)

    assert torch.allclose(
        norms,
        torch.ones(4),
        atol=1e-6,
    )
def test_full_softmax_loss_is_finite():
    from models.two_tower import full_softmax_loss

    user_vectors = torch.randn(4, 32)
    movie_vectors = torch.randn(10, 32)
    positive_movie_idx = torch.tensor([0, 1, 2, 3])

    loss = full_softmax_loss(
        user_vectors,
        movie_vectors,
        positive_movie_idx,
        tau=0.07,
    )

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_inbatch_loss_is_finite():
    from models.two_tower import inbatch_loss

    user_vectors = torch.randn(4, 32)
    movie_vectors = torch.randn(4, 32)

    user_idx = torch.tensor([0, 1, 2, 3])
    movie_idx = torch.tensor([0, 1, 2, 3])

    train_pos_mask = torch.zeros(10, 10, dtype=torch.bool)
    log_item_prob = torch.zeros(10)

    loss = inbatch_loss(
        user_vectors,
        movie_vectors,
        user_idx,
        movie_idx,
        train_pos_mask,
        log_item_prob,
        tau=0.07,
    )

    assert torch.isfinite(loss)
    assert loss.ndim == 0