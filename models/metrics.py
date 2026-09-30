"""Shared eval harness -- use the SAME functions for MF, popularity,
random, and the two-tower model so numbers are actually comparable."""
import torch


def to_mask(df, n_users, n_movies, id_col_user="user_idx", id_col_movie="movie_idx"):
    """Boolean [n_users, n_movies], True where that (user, movie) occurs in df.
    Rows with an unmapped (NaN) user_idx/movie_idx -- e.g. a movie that
    fell outside the fit_train vocabulary -- are dropped: they can't
    reference a real row/column in the matrix, and silently casting NaN
    to int64 produces garbage indices rather than an error."""
    known = df[[id_col_user, id_col_movie]].notna().all(axis=1)
    df = df.loc[known]
    m = torch.zeros(n_users, n_movies, dtype=torch.bool)
    u = torch.as_tensor(df[id_col_user].to_numpy(), dtype=torch.long)
    v = torch.as_tensor(df[id_col_movie].to_numpy(), dtype=torch.long)
    m[u, v] = True
    return m


@torch.no_grad()
def rank_metrics(scores, seen_mask, relevant_mask, ks=(10, 20)):
    """scores: [U, M] similarity matrix.
    seen_mask: [U, M] bool, items to exclude from ranking (already interacted).
    relevant_mask: [U, M] bool, held-out positives to check against.
    Users with an empty relevant row are skipped (matches mf.py's behavior
    of only counting users who have at least one *known* relevant movie)."""
    scores = scores.masked_fill(seen_mask, float("-inf"))
    K = max(ks)
    topk = scores.topk(K, dim=1).indices                       # [U, K]
    hit = relevant_mask.gather(1, topk).float()                # [U, K]
    n_rel = relevant_mask.sum(1).float()
    valid = n_rel > 0

    out = {"users_evaluated": int(valid.sum().item())}
    for k in ks:
        h = hit[:, :k]
        disc = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32))
        dcg = (h * disc).sum(1)
        ideal_n = n_rel.clamp(max=k).long()
        idcg = torch.cat([torch.zeros(1), disc.cumsum(0)])[ideal_n]
        out[f"hit@{k}"] = (h.sum(1) > 0).float()[valid].mean().item()
        out[f"recall@{k}"] = (h.sum(1) / n_rel.clamp(min=1))[valid].mean().item()
        out[f"ndcg@{k}"] = (dcg / idcg.clamp(min=1e-9))[valid].mean().item()

    has_hit = hit.sum(1) > 0
    first_rank = hit.argmax(1).float() + 1
    out[f"mrr@{K}"] = torch.where(
        has_hit, 1.0 / first_rank, torch.zeros_like(first_rank)
    )[valid].mean().item()
    out[f"coverage@{ks[0]}"] = topk[:, :ks[0]].unique().numel() / scores.size(1)
    return out


def relevant_mask_filtered(df, movie_to_idx, n_users, n_movies, user_to_idx):
    """Build the relevant-items mask the way mf.py does: rating>=4 AND the
    movie must be in movie_to_idx (known catalog). This is the piece the
    two-tower evaluate.py was missing -- without it, a user whose only
    test positive is an unseen movie is still counted as 'evaluated' but
    can never be hit, which deflates the two-tower number relative to MF."""
    pos = df[df["rating"] >= 4].copy()
    pos = pos[pos["movie_id"].isin(movie_to_idx)]
    pos["user_idx"] = pos["user_id"].map(user_to_idx)
    pos["movie_idx"] = pos["movie_id"].map(movie_to_idx)
    return to_mask(pos, n_users, n_movies)


def popularity_scores(positive_fit_train_df, n_users, n_movies):
    counts = torch.bincount(
        torch.as_tensor(positive_fit_train_df["movie_idx"].to_numpy(), dtype=torch.long),
        minlength=n_movies,
    ).float()
    return counts.unsqueeze(0).expand(n_users, -1).clone()


def random_scores(n_users, n_movies, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_users, n_movies, generator=g)


@torch.no_grad()
def score_matrix(user_tower, movie_tower, user_features_t, movie_features_t):
    user_tower.eval()
    movie_tower.eval()
    u_ids = torch.arange(user_features_t.size(0))
    m_ids = torch.arange(movie_features_t.size(0))
    u = user_tower(u_ids, user_features_t)
    m = movie_tower(m_ids, movie_features_t)
    return u @ m.T