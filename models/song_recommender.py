import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings


class HindiSongRecommender:

    def __init__(self, data_path="data/songs.csv"):
        self.df = pd.read_csv(data_path)

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.df["combined_text"] = (
            self.df["title"] + " " +
            self.df["artist"] + " " +
            self.df["lyrics"]
        )

        self.embeddings = self.embedding_model.embed_documents(
            self.df["combined_text"].tolist()
        )

    def recommend(self, query, top_n=5):
        query_embedding = self.embedding_model.embed_query(query)

        scores = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]

        self.df["similarity"] = scores
        return self.df.sort_values("similarity", ascending=False).head(top_n)

    def recommend_by_mood(self, mood, top_n=5):
        filtered = self.df[self.df["mood"] == mood]
        return filtered.head(top_n)
