import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Movie Recommender")
st.caption("Personalized recommendations, powered by a two-tower retrieval model.")

with st.sidebar:
    st.header("Model performance")
    st.caption(
        "Same 1,582-movie vocabulary and seen/relevant masks for every "
        "model below -- an apples-to-apples comparison, not just a "
        "leaderboard number."
    )
    st.table(
        {
            "Model": ["Random", "Popularity", "MF (fair)", "Two-Tower"],
            "Hit@10": ["8.5%", "39.3%", "27.7%", "56.6%"],
            "NDCG@10": ["0.010", "0.092", "0.055", "0.146"],
            "Coverage@10": ["99.7%", "4.7%", "16.2%", "44.6%"],
        }
    )
    st.caption(
        "Coverage@10 = fraction of the catalog that ever appears in a "
        "top-10 list across all users -- a check against the model "
        "just recommending whatever's popular."
    )

col1, col2 = st.columns([2, 1])
with col1:
    user_id = st.number_input("User ID (1-943)", min_value=1, max_value=943, value=1, step=1)
with col2:
    k = st.slider("Recommendations", min_value=1, max_value=20, value=10)

if st.button("Get recommendations", type="primary"):
    try:
        response = requests.post(
            f"{API_URL}/recommend",
            json={"user_id": int(user_id), "k": int(k)},
            timeout=5,
        )
    except requests.exceptions.ConnectionError:
        st.error(
            "Couldn't reach the API. Make sure it's running: "
            "`uvicorn app.main:app --reload`"
        )
    else:
        if response.status_code == 404:
            st.warning(response.json()["detail"])
        elif response.status_code != 200:
            st.error(f"API error {response.status_code}: {response.text}")
        else:
            data = response.json()
            st.subheader(f"Recommended for user {data['user_id']}")
            for rec in data["recommendations"]:
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"**{rec['title']}**")
                    if rec["genres"]:
                        st.caption(", ".join(rec["genres"]))
                with right:
                    st.markdown(f"⭐ {rec['score']:.3f}")
                st.divider()