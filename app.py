import streamlit as st
from models.song_recommender import HindiSongRecommender

st.set_page_config(page_title="Hindi Song Recommender")

st.title("🎵 Hindi Song Recommendation System")

recommender = HindiSongRecommender()

option = st.selectbox(
    "Choose Recommendation Type",
    ["Lyrics / Song Name", "Mood"]
)

if option == "Lyrics / Song Name":
    query = st.text_input("Enter song name or lyrics")
    top_n = st.slider("Number of recommendations", 1, 10, 5)

    if st.button("Recommend"):
        if query.strip() == "":
            st.warning("Please enter something!")
        else:
            results = recommender.recommend(query, top_n)
            st.dataframe(results[["title", "artist", "mood"]])

else:
    mood = st.selectbox("Select Mood", ["sad", "romantic", "happy"])
    if st.button("Recommend"):
        results = recommender.recommend_by_mood(mood)
        st.dataframe(results[["title", "artist", "mood"]])
