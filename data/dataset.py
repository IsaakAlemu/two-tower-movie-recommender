import torch
from torch.utils.data import Dataset



class MovieLensDataset(Dataset):
    def __init__(self, interactions, user_features, movie_features):
        self.interactions = interactions.reset_index(drop=True)
        self.user_features = user_features
        self.movie_features = movie_features

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]

        user_idx = row["user_idx"]
        movie_idx = row["movie_idx"]

        user_features = self.user_features.iloc[int(user_idx)].to_numpy()
        movie_features = self.movie_features.iloc[int(movie_idx)].to_numpy()

        return {
            "user_idx": torch.tensor(
                user_idx,
                dtype=torch.long
            ),
            "movie_idx": torch.tensor(
                movie_idx,
                dtype=torch.long
            ),
            "user_features": torch.tensor(
                user_features,
                dtype=torch.float32
            ),
            "movie_features": torch.tensor(
                movie_features,
                dtype=torch.float32
            )
        }
  