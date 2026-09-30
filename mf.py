import pandas as pd
import numpy as np

np.random.seed(42)

ratings = pd.read_csv(
    "data/raw/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# Sort exactly the same way as our locked temporal split
ratings = ratings.sort_values(
    ["user_id", "timestamp", "movie_id"]
)

# Create the same 80/20 temporal split
test_parts = []

for user_id, user_ratings in ratings.groupby("user_id"):
    n_test = max(1, int(len(user_ratings) * 0.2))
    test_parts.append(user_ratings.tail(n_test))

test = pd.concat(test_parts)
train = ratings.drop(test.index)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

print("Train:", train.shape)
print("Test:", test.shape)


# Get unique users and movies from the training data
user_ids = train["user_id"].unique()
movie_ids = train["movie_id"].unique()

# Map original IDs to array indices
user_to_idx = {
    user_id: idx
    for idx, user_id in enumerate(user_ids)
}

movie_to_idx = {
    movie_id: idx
    for idx, movie_id in enumerate(movie_ids)
}

print("\nNumber of users:", len(user_to_idx))
print("Number of movies:", len(movie_to_idx))


# Movies that appear in the test set but not in training
unseen_test_movies = set(test["movie_id"]) - set(train["movie_id"])

print("\nMovies unseen during training:", len(unseen_test_movies))

unseen_test_ratings = test[
    test["movie_id"].isin(unseen_test_movies)
]

print(
    "Test ratings for unseen movies:",
    len(unseen_test_ratings)
)


# Number of latent factors
k = 10

# Initialize user and movie latent vectors
U = np.random.normal(
    0, 0.1, size=(len(user_to_idx), k)
)

V = np.random.normal(
    0, 0.1, size=(len(movie_to_idx), k)
)


# Global average rating
global_mean = train["rating"].mean()

# User and movie bias terms
user_bias = np.zeros(len(user_to_idx))
movie_bias = np.zeros(len(movie_to_idx))

print("\nGlobal mean rating:", global_mean)
print("User bias shape:", user_bias.shape)
print("Movie bias shape:", movie_bias.shape)

print("\nUser matrix shape:", U.shape)
print("Movie matrix shape:", V.shape)


# Prediction function
def predict(user_idx, movie_idx):
    return (
        global_mean
        + user_bias[user_idx]
        + movie_bias[movie_idx]
        + np.dot(U[user_idx], V[movie_idx])
    )


# Check initial prediction
example_user = train.iloc[0]["user_id"]
example_movie = train.iloc[0]["movie_id"]
actual_rating = train.iloc[0]["rating"]

user_idx = user_to_idx[example_user]
movie_idx = movie_to_idx[example_movie]

prediction = predict(user_idx, movie_idx)

print("\nExample:")
print("User:", example_user)
print("Movie:", example_movie)
print("Actual rating:", actual_rating)
print("Initial prediction:", prediction)


# Training settings
learning_rate = 0.01
lambda_reg = 0.02
epochs = 20

loss_history = []


# Training loop
for epoch in range(epochs):

    # Update U, V, user_bias and movie_bias
    for _, row in train.iterrows():

        user_idx = user_to_idx[row["user_id"]]
        movie_idx = movie_to_idx[row["movie_id"]]
        rating = row["rating"]

        prediction = predict(user_idx, movie_idx)

        error = rating - prediction

        # Save current vectors before updating
        user_vector = U[user_idx].copy()
        movie_vector = V[movie_idx].copy()

        # Update latent vectors
        U[user_idx] += learning_rate * (
            error * movie_vector
            - lambda_reg * user_vector
        )

        V[movie_idx] += learning_rate * (
            error * user_vector
            - lambda_reg * movie_vector
        )

        # Update user bias
        user_bias[user_idx] += learning_rate * (
            error
            - lambda_reg * user_bias[user_idx]
        )

        # Update movie bias
        movie_bias[movie_idx] += learning_rate * (
            error
            - lambda_reg * movie_bias[movie_idx]
        )

    # Calculate training MSE after this epoch
    squared_error = 0.0

    for _, row in train.iterrows():

        user_idx = user_to_idx[row["user_id"]]
        movie_idx = movie_to_idx[row["movie_id"]]
        rating = row["rating"]

        prediction = predict(user_idx, movie_idx)

        error = rating - prediction

        squared_error += error ** 2

    mse = squared_error / len(train)

    loss_history.append(mse)

    print(
        f"Epoch {epoch + 1}/{epochs} - "
        f"Training MSE: {mse:.4f}"
    )


# --------------------------------------------------
# Test RMSE
# --------------------------------------------------

squared_error = 0.0
evaluated = 0

for _, row in test.iterrows():

    user_id = row["user_id"]
    movie_id = row["movie_id"]
    rating = row["rating"]

    # Skip movies never seen during training
    if movie_id not in movie_to_idx:
        continue

    user_idx = user_to_idx[user_id]
    movie_idx = movie_to_idx[movie_id]

    prediction = predict(user_idx, movie_idx)

    error = rating - prediction

    squared_error += error ** 2
    evaluated += 1


rmse = np.sqrt(squared_error / evaluated)

print("\nMF Test Evaluation")
print("------------------")
print("Test ratings:", len(test))
print("Evaluated ratings:", evaluated)
print("Skipped ratings:", len(test) - evaluated)
print("Coverage:", evaluated / len(test) * 100, "%")
print("Test RMSE:", rmse)


# Hit@10 using only relevant test ratings (rating >= 4)
hits = 0
users_evaluated = 0

for user_id, user_test in test.groupby("user_id"):

    # Only ratings >= 4 are considered relevant
    relevant_movies = set(
        user_test.loc[user_test["rating"] >= 4, "movie_id"]
    )

    # Ignore users with no relevant test movie
    if not relevant_movies:
        continue

    # Keep only movies that MF knows from training
    relevant_movies = {
        movie_id
        for movie_id in relevant_movies
        if movie_id in movie_to_idx
    }

    if not relevant_movies:
        continue

    users_evaluated += 1

    # Movies known to MF and not seen by this user during training
    train_movies = set(
        train.loc[train["user_id"] == user_id, "movie_id"]
    )

    candidate_movies = [
        movie_id
        for movie_id in movie_to_idx
        if movie_id not in train_movies
    ]

    user_idx = user_to_idx[user_id]

    predictions = []

    for movie_id in candidate_movies:
        movie_idx = movie_to_idx[movie_id]
        score = predict(user_idx, movie_idx)
        predictions.append((movie_id, score))

    predictions.sort(key=lambda x: x[1], reverse=True)

    top_10 = {
        movie_id
        for movie_id, score in predictions[:10]
    }

    if top_10 & relevant_movies:
        hits += 1


hit_at_10 = hits / users_evaluated

print(f"Users evaluated for Hit@10: {users_evaluated}")
print(f"Users with a Hit@10: {hits}")
print(f"Hit@10: {hit_at_10:.4f} ({hit_at_10 * 100:.2f}%)")