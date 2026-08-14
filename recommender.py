"""MovieLens recommender: content, collaborative (SVD), and hybrid ranking.

Run with: python recommender.py --user-id 1 --method hybrid
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


class MovieRecommender:
    def __init__(self, data_dir: str | Path = "data", random_state: int = 42):
        self.data_dir = Path(data_dir)
        self.random_state = random_state

    def load_data(self) -> "MovieRecommender":
        """Load MovieLens latest-small, downloading it once if it is absent."""
        movies_file = self.data_dir / "ml-latest-small" / "movies.csv"
        ratings_file = self.data_dir / "ml-latest-small" / "ratings.csv"
        if not movies_file.exists() or not ratings_file.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            print("Downloading MovieLens latest-small...")
            with urlopen(DATA_URL) as response:
                with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                    archive.extractall(self.data_dir)
        self.movies = pd.read_csv(movies_file)
        self.ratings = pd.read_csv(ratings_file)
        return self

    def fit_content_model(self) -> "MovieRecommender":
        """Represent genres with TF-IDF and compare films using cosine similarity."""
        text = self.movies["genres"].fillna("").str.replace("|", " ", regex=False)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.content_matrix = self.vectorizer.fit_transform(text)
        self.movie_positions = pd.Series(self.movies.index, index=self.movies.movieId)
        return self

    def fit_collaborative_model(self, factors: int = 50, epochs: int = 25,
                                learning_rate: float = 0.01, regularization: float = 0.05) -> "MovieRecommender":
        """Train biased matrix factorization with SGD on explicit star ratings."""
        self.user_ids = np.sort(self.ratings.userId.unique())
        self.item_ids = np.sort(self.ratings.movieId.unique())
        self.user_index = {value: index for index, value in enumerate(self.user_ids)}
        self.item_index = {value: index for index, value in enumerate(self.item_ids)}
        rng = np.random.default_rng(self.random_state)
        self.global_mean = float(self.ratings.rating.mean())
        self.user_bias = np.zeros(len(self.user_ids))
        self.item_bias = np.zeros(len(self.item_ids))
        self.user_factors = rng.normal(0, 0.1, (len(self.user_ids), factors))
        self.item_factors = rng.normal(0, 0.1, (len(self.item_ids), factors))
        observations = self.ratings[["userId", "movieId", "rating"]].to_numpy()
        for _ in range(epochs):
            rng.shuffle(observations)
            for user_id, movie_id, rating in observations:
                u, i = self.user_index[user_id], self.item_index[movie_id]
                prediction = self.global_mean + self.user_bias[u] + self.item_bias[i] + self.user_factors[u] @ self.item_factors[i]
                error = rating - prediction
                self.user_bias[u] += learning_rate * (error - regularization * self.user_bias[u])
                self.item_bias[i] += learning_rate * (error - regularization * self.item_bias[i])
                user_vector = self.user_factors[u].copy()
                self.user_factors[u] += learning_rate * (error * self.item_factors[i] - regularization * self.user_factors[u])
                self.item_factors[i] += learning_rate * (error * user_vector - regularization * self.item_factors[i])
        return self

    def similar_movies(self, title: str, n: int = 10) -> pd.DataFrame:
        """Content-only recommendations for a title (case-insensitive substring match)."""
        matches = self.movies[self.movies.title.str.contains(title, case=False, regex=False, na=False)]
        if matches.empty:
            raise ValueError(f"No movie matching {title!r} was found.")
        source = matches.iloc[0]
        pos = self.movie_positions[source.movieId]
        scores = cosine_similarity(self.content_matrix[pos], self.content_matrix).ravel()
        candidates = self.movies.assign(content_score=scores).query("movieId != @source.movieId")
        return candidates.nlargest(n, "content_score")[["movieId", "title", "genres", "content_score"]]

    def _content_scores_for_user(self, user_id: int) -> pd.Series:
        liked = self.ratings[(self.ratings.userId == user_id) & (self.ratings.rating >= 4.0)]
        if liked.empty:
            return pd.Series(0.0, index=self.movies.movieId)
        positions = [self.movie_positions[mid] for mid in liked.movieId if mid in self.movie_positions]
        profile = self.content_matrix[positions].mean(axis=0)
        scores = cosine_similarity(profile, self.content_matrix).ravel()
        return pd.Series(scores, index=self.movies.movieId)

    @staticmethod
    def _minmax(values: pd.Series) -> pd.Series:
        spread = values.max() - values.min()
        return (values - values.min()) / spread if spread else pd.Series(0.0, index=values.index)

    def recommend_for_user(self, user_id: int, method: str = "hybrid", n: int = 10,
                           content_weight: float = 0.4) -> pd.DataFrame:
        """Recommend unseen movies. Hybrid score blends normalised content and SVD scores."""
        if user_id not in self.user_index:
            raise ValueError(f"Unknown user {user_id}. Choose a user in the MovieLens ratings data.")
        watched = set(self.ratings.loc[self.ratings.userId == user_id, "movieId"])
        candidates = self.movies[~self.movies.movieId.isin(watched)].copy()
        content = self._content_scores_for_user(user_id).reindex(candidates.movieId).fillna(0.0)
        u = self.user_index[user_id]
        collaborative = pd.Series(
            [self.global_mean + self.user_bias[u] + self.item_bias[self.item_index[mid]] + self.user_factors[u] @ self.item_factors[self.item_index[mid]] if mid in self.item_index else self.global_mean for mid in candidates.movieId],
            index=candidates.movieId,
        )
        if method == "content":
            score = content
        elif method == "collaborative":
            score = collaborative
        elif method == "hybrid":
            score = content_weight * self._minmax(content) + (1 - content_weight) * self._minmax(collaborative)
        else:
            raise ValueError("method must be content, collaborative, or hybrid")
        candidates["score"] = candidates.movieId.map(score)
        return candidates.nlargest(n, "score")[["movieId", "title", "genres", "score"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="MovieLens recommendation demo")
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--method", choices=["content", "collaborative", "hybrid"], default="hybrid")
    parser.add_argument("--title", help="Return content-similar movies for this title instead of user recommendations")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--content-weight", type=float, default=0.4)
    args = parser.parse_args()
    model = MovieRecommender().load_data().fit_content_model().fit_collaborative_model()
    if args.title:
        result = model.similar_movies(args.title, args.top)
    else:
        result = model.recommend_for_user(args.user_id, args.method, args.top, args.content_weight)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
