# Movie Recommendation System (AI + Personalization)

A compact Netflix-style MovieLens recommender implementing all three requested approaches:

| Approach | Implementation | Use |
| --- | --- | --- |
| Content-based filtering | TF-IDF genre embeddings + cosine similarity | Similar titles and cold-start preferences |
| Collaborative filtering | Biased matrix factorization trained with SGD | Learns taste patterns from user ratings |
| Hybrid model | Weighted, normalized blend of both scores | Default personalized ranking |

## Quick start

```powershell
cd C:\Users\keert\Documents\Codex\2026-07-31\movie-recommendation-system-ai-personalization-what\outputs\movie-recommender
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python recommender.py --user-id 1 --method hybrid --top 10
```

On first run, the program retrieves MovieLens `latest-small` from GroupLens into `data/`. If downloading is not possible, place `movies.csv` and `ratings.csv` at `data/ml-latest-small/`.

## Examples

```powershell
# Content recommendations for a selected title
python recommender.py --title "Toy Story" --top 5

# Collaborative recommendations based only on other users' ratings
python recommender.py --user-id 25 --method collaborative

# Hybrid: increase the genre/content component from 40% to 60%
python recommender.py --user-id 25 --method hybrid --content-weight 0.6
```

## How personalization works

For content filtering, a user's profile is the average TF-IDF vector of films they rated 4 stars or above. Cosine similarity measures how close each unseen movie is to that profile.

For collaborative filtering, each user and movie receives a learned latent vector. The predicted rating is:

`global mean + user bias + movie bias + user-vector · movie-vector`

The hybrid model min-max normalizes the content and collaborative scores before blending them. Movies a user has already rated are always excluded.

## Scaling notes

This implementation is deliberately transparent and suitable for MovieLens `latest-small`. At production scale, use an implicit-feedback library such as `implicit` (ALS) for viewing/click data, store embeddings in a feature store, and add a candidate-generation and re-ranking pipeline. `scikit-surprise` can also replace the included SGD model with `SVD` for an explicit-feedback baseline.
