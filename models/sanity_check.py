from data.preprocess import movies, genre_names, train, user_to_idx
from models.build_index import build_index, recommend


id_to_title = dict(zip(movies["movie_id"], movies["title"]))


def genres_of(movie_id):
    row = movies.loc[movies["movie_id"] == movie_id]
    if row.empty:
        return ""
    row = row.iloc[0]
    return ", ".join(g for g in genre_names if row[g] == 1)


def show_user(user_id, index, user_vectors_np, k=10):
    print(f"\n{'='*60}\nUser {user_id}")
    print(f"{'='*60}")

    print("\nRated >=4 in fit_train/val (what they actually liked):")
    history = (
        train[(train["user_id"] == user_id) & (train["rating"] >= 4)]
        .sort_values("rating", ascending=False)
    )
    for _, row in history.iterrows():
        title = id_to_title.get(row["movie_id"], row["movie_id"])
        print(f"  {row['rating']}  {title}  [{genres_of(row['movie_id'])}]")

    print(f"\nTop-{k} FAISS recommendations:")
    for movie_id, score in recommend(user_id, k=k, index=index, user_vectors_np=user_vectors_np):
        title = id_to_title.get(movie_id, movie_id)
        print(f"  {score:.4f}  {title}  [{genres_of(movie_id)}]")


if __name__ == "__main__":
    index, user_vectors_np = build_index()

    # Check a few different users, not just one -- a single user's list
    # coinciding by chance tells you less than three or four agreeing.
    sample_users = list(user_to_idx.keys())[:4]
    for user_id in sample_users:
        show_user(user_id, index, user_vectors_np)