import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    """ID embedding concatenated with side features, through a 2-layer
    MLP, L2-normalized. No activation on the output layer -- ReLU there
    would confine every vector to the positive orthant and crush cosine
    similarity toward a narrow, high band."""

    def __init__(self, n_ids, feat_dim, id_dim=32, hidden=64, out_dim=32):
        super().__init__()
        self.id_emb = nn.Embedding(n_ids, id_dim)
        # Default nn.Embedding init is N(0,1) -> ID vectors have norm ~5.6,
        # which drowns out 0/1-valued side features at the concat. Shrink it.
        nn.init.normal_(self.id_emb.weight, std=0.1)
        self.mlp = nn.Sequential(
            nn.Linear(id_dim + feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, ids, feats):
        x = torch.cat([self.id_emb(ids), feats], dim=-1)
        return F.normalize(self.mlp(x), p=2, dim=-1)


class UserTower(Tower):
    def __init__(self, num_users, feat_dim=27, id_dim=32, out_dim=32):
        # feat_dim=27: 6 one-hot age buckets + 21 one-hot occupations
        super().__init__(num_users, feat_dim, id_dim=id_dim, out_dim=out_dim)


class MovieTower(Tower):
    def __init__(self, num_movies, feat_dim=20, id_dim=32, out_dim=32):
        # feat_dim=20: 19 genre multi-hot + 1 normalized release year
        super().__init__(num_movies, feat_dim, id_dim=id_dim, out_dim=out_dim)


def full_softmax_loss(user_vectors, all_movie_vectors, pos_movie_idx, tau):
    """Exact softmax over the WHOLE catalog. With only 1,612 movies this
    is cheap and has no sampling bias -- no logQ correction needed, no
    in-batch false-negative masking needed. Use this unless/until you
    have a catalog too large to score in full each step."""
    logits = user_vectors @ all_movie_vectors.T / tau        # [B, M]
    return F.cross_entropy(logits, pos_movie_idx)


def inbatch_loss(user_vectors, movie_vectors, user_idx, movie_idx,
                  train_pos_mask, log_item_prob, tau):
    """In-batch sampled softmax, corrected two ways:
    1) logQ: subtract the log sampling probability of each candidate
       column's item (items appear as negatives in proportion to how
       often they're positives in the batch -- popular movies are
       over-sampled as negatives unless corrected for).
    2) mask every off-diagonal (user_i, movie_j) pair that is itself a
       known training positive (same user liked it elsewhere in the
       batch, or movie_j duplicates movie_i) -- otherwise you penalize
       the model for correctly ranking a true positive highly."""
    B = user_vectors.size(0)
    logits = user_vectors @ movie_vectors.T / tau             # [B, B]
    logits = logits - log_item_prob[movie_idx].unsqueeze(0)
    known_pos = train_pos_mask[user_idx.unsqueeze(1), movie_idx.unsqueeze(0)]
    eye = torch.eye(B, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(known_pos & ~eye, float("-inf"))
    labels = torch.arange(B, device=logits.device)
    return F.cross_entropy(logits, labels)