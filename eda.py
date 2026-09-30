import pandas as pd

# Load MovieLens ratings
ratings = pd.read_csv(
    "data/raw/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# Count how many ratings each movie has
ratings_per_movie = ratings.groupby("movie_id").size()

# Summary statistics
print("Ratings per movie:")
print(ratings_per_movie.describe())

# Additional useful statistics
print("\nMovies with < 5 ratings:", (ratings_per_movie < 5).sum())
print("Movies with < 10 ratings:", (ratings_per_movie < 10).sum())
print("Movies with < 20 ratings:", (ratings_per_movie < 20).sum())
print("Movies with >= 50 ratings:", (ratings_per_movie >= 50).sum())
print("Movies with >= 100 ratings:", (ratings_per_movie >= 100).sum())

# 10 most-rated movies
print("\nMost-rated movies:")
print(ratings_per_movie.sort_values(ascending=False).head(10))

# Sort all ratings chronologically for each user
ratings = ratings.sort_values(
    ["user_id", "timestamp", "movie_id"]
)

# Take the last 20% of each user's ratings as test data
test_parts = []

for user_id, user_ratings in ratings.groupby("user_id"):
    n_test = max(1, int(len(user_ratings) * 0.2))

    test_parts.append(user_ratings.tail(n_test))

# Combine all users' test ratings
test = pd.concat(test_parts)

# Everything not in the test set becomes training data
train = ratings.drop(test.index)

# Reset indexes
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Check the result
print("\nTrain shape:", train.shape)
print("Test shape:", test.shape)

print("\nTrain ratings:", len(train))
print("Test ratings:", len(test))

print("\nOverall test percentage:", len(test) / len(ratings) * 100)


# Verify temporal ordering more carefully
train_latest = train.groupby("user_id")["timestamp"].max()
test_earliest = test.groupby("user_id")["timestamp"].min()

# Compare the two boundary timestamps
boundary_check = pd.DataFrame({
    "train_latest": train_latest,
    "test_earliest": test_earliest
})

print("\nTemporal split verification:")
print("Users checked:", len(boundary_check))

print(
    "Users where train ends BEFORE test:",
    (boundary_check["train_latest"] < boundary_check["test_earliest"]).sum()
)

print(
    "Users where train/test boundary has SAME timestamp:",
    (boundary_check["train_latest"] == boundary_check["test_earliest"]).sum()
)

print(
    "Users where train ends AFTER test begins:",
    (boundary_check["train_latest"] > boundary_check["test_earliest"]).sum()
)