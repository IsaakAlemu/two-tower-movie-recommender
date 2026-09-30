import pandas as pd


# ============================================
# 1. Load ratings
# ============================================

ratings = pd.read_csv(
    "data/raw/ml-100k/u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

ratings = ratings.sort_values(
    ["user_id", "timestamp", "movie_id"]
)


# ============================================
# 2. Per-user temporal TEST split (final holdout, never touched again
#    until the very end)
# ============================================

test_parts = []

for user_id, user_ratings in ratings.groupby("user_id"):
    n_test = max(1, int(len(user_ratings) * 0.2))
    test_parts.append(user_ratings.tail(n_test))

test = pd.concat(test_parts).reset_index(drop=True)

# `train` = the 80% development pool (fit_train + val). Kept under this
# name because evaluate.py uses it as the "seen" mask at test time --
# everything the model could have been exposed to before test.
train = ratings.drop(pd.concat(test_parts).index).reset_index(drop=True)

print("Train (dev pool):", train.shape)
print("Test:", test.shape)


# ============================================
# 3. Per-user temporal VALIDATION split, carved out of `train` only.
#    fit_train is what's left -- this is the ONLY data the model or
#    its vocabulary is allowed to be built from.
# ============================================

val_parts = []

for user_id, user_ratings in train.groupby("user_id"):
    n_val = max(1, int(len(user_ratings) * 0.1))
    val_parts.append(user_ratings.tail(n_val))

val = pd.concat(val_parts).reset_index(drop=True)
fit_train = train.drop(pd.concat(val_parts).index).reset_index(drop=True)

print("Fit-train:", fit_train.shape)
print("Val:", val.shape)
print(
    "Fit-train/val overlap:",
    len(set(fit_train.index) & set(val.index))
)


# ============================================
# 4. Create ID mappings from FIT_TRAIN ONLY
#    (previously this used `train`, which quietly included val movies)
# ============================================

user_ids = fit_train["user_id"].unique()

user_to_idx = {
    user_id: idx
    for idx, user_id in enumerate(user_ids)
}

movie_ids = fit_train["movie_id"].unique()

movie_to_idx = {
    movie_id: idx
    for idx, movie_id in enumerate(movie_ids)
}

print("\nFit-train users:", len(user_to_idx))
print("Fit-train movies:", len(movie_to_idx))


# ============================================
# 5. Attach indices to every split against the fit_train vocabulary.
#    val and test can now both contain movies/users with NaN movie_idx
#    -- that's expected and correct, not a bug.
# ============================================

for df in (fit_train, val, train, test):
    df["user_idx"] = df["user_id"].map(user_to_idx)
    df["movie_idx"] = df["movie_id"].map(movie_to_idx)

print(
    "\nUnknown fit-train users:", fit_train["user_idx"].isna().sum(),
    "| movies:", fit_train["movie_idx"].isna().sum()
)

unseen_val = val[val["movie_idx"].isna()]
print(
    "Unseen val movie interactions:", len(unseen_val),
    "| unique unseen val movies:", unseen_val["movie_id"].nunique()
)

unseen_test = test[test["movie_idx"].isna()]
print(
    "Unseen test movie interactions:", len(unseen_test),
    "| unique unseen test movies:", unseen_test["movie_id"].nunique()
)


# ============================================
# 6. Positive interactions used for training gradients
# ============================================

positive_fit_train = fit_train[fit_train["rating"] >= 4].copy()

print(
    "\nPositive fit-train interactions:",
    positive_fit_train.shape
)
print(
    positive_fit_train["rating"]
    .value_counts()
    .sort_index()
)

# Kept for an OPTIONAL final retrain after model selection is done, using
# all of train's (fit_train + val) positives. NOTE: some rows here may
# have NaN movie_idx (movies that only appeared in val, never in
# fit_train) -- drop those before using this for anything.
positive_train = train[train["rating"] >= 4].copy()


# ============================================
# 7. Load and encode user metadata (age buckets, occupation)
# ============================================

users = pd.read_csv(
    "data/raw/ml-100k/u.user",
    sep="|",
    names=["user_id", "age", "gender", "occupation", "zip"]
)

bins = [0, 17, 24, 34, 44, 54, float("inf")]
labels = [0, 1, 2, 3, 4, 5]

users["age_bucket"] = pd.cut(
    users["age"], bins=bins, labels=labels
).astype(int)

users["user_idx"] = users["user_id"].map(user_to_idx)

print(
    "\nUsers without a fit-train index:",
    users["user_idx"].isna().sum()
)

# Keep only users that appeared in fit_train (should be all 943 -- the
# per-user split guarantees every user has interactions in fit_train).
users = users.dropna(subset=["user_idx"]).copy()
users["user_idx"] = users["user_idx"].astype(int)

users = users.sort_values("user_idx").reset_index(drop=True)

users["age_bucket"] = pd.Categorical(
    users["age_bucket"], categories=[0, 1, 2, 3, 4, 5]
)
age_bucket_features = pd.get_dummies(
    users["age_bucket"], prefix="age", dtype=int
)

occupation_features = pd.get_dummies(
    users["occupation"], dtype=int
)

user_features = pd.concat(
    [age_bucket_features, occupation_features], axis=1
)

assert (
    users["user_idx"].to_numpy() == user_features.index.to_numpy()
).all()

print("User feature shape:", user_features.shape)
print("Missing user feature values:", user_features.isna().sum().sum())


# ============================================
# 8. Load and encode movie metadata (genres, release year)
# ============================================

movies = pd.read_csv(
    "data/raw/ml-100k/u.item",
    sep="|",
    encoding="latin-1",
    header=None
)

genres = pd.read_csv("data/raw/ml-100k/u.genre", sep="|", header=None)
genre_names = genres[0].tolist()

movies.columns = [
    "movie_id", "title", "release_date",
    "video_release_date", "imdb_url"
] + genre_names

movies["release_year"] = pd.to_datetime(
    movies["release_date"]
).dt.year

median_year = movies["release_year"].median()
movies["release_year"] = movies["release_year"].fillna(median_year)

min_year, max_year = 1922, 1998
movies["release_year_norm"] = (
    (movies["release_year"] - min_year) / (max_year - min_year)
)

movies["movie_idx"] = movies["movie_id"].map(movie_to_idx)

print(
    "\nMovies without a fit-train index:",
    movies["movie_idx"].isna().sum()
)

movies_train = movies.dropna(subset=["movie_idx"]).copy()
movies_train["movie_idx"] = movies_train["movie_idx"].astype(int)
movies_train = movies_train.sort_values("movie_idx").reset_index(drop=True)

movie_features = movies_train[genre_names + ["release_year_norm"]]

print("Movie feature shape:", movie_features.shape)
print("Missing movie feature values:", movie_features.isna().sum().sum())


# ============================================
# 9. Final verification
# ============================================

print("\nFinal verification")
print("------------------")
print("Fit-train rows:", len(fit_train))
print("Val rows:", len(val))
print("Test rows:", len(test))
print("Positive fit-train rows:", len(positive_fit_train))
print("Fit-train users:", len(user_to_idx))
print("Fit-train movies:", len(movie_to_idx))
print("User features:", user_features.shape)
print("Movie features:", movie_features.shape)
print(
    "Unseen val movie interactions:", len(unseen_val),
    "| unique:", unseen_val["movie_id"].nunique()
)
print(
    "Unseen test movie interactions:", len(unseen_test),
    "| unique:", unseen_test["movie_id"].nunique()
)