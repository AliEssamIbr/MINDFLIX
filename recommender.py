import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os

# ─────────────────────────────────────────
# LOAD DATA  (cached — runs once)
# ─────────────────────────────────────────
ratings_path = "data/ratings.csv"
movies_path  = "data/movies.csv"

ratings = pd.read_csv(ratings_path)
movies  = pd.read_csv(movies_path)
data    = pd.merge(ratings, movies, on="movieId")

global_avg_ratings = data.groupby("movieId")["rating"].mean().to_dict()

user_ids  = data["userId"].unique()
movie_ids = data["movieId"].unique()

user_map  = {uid: i for i, uid in enumerate(user_ids)}
movie_map = {mid: i for i, mid in enumerate(movie_ids)}

data["user"]  = data["userId"].map(user_map)
data["movie"] = data["movieId"].map(movie_map)

num_users  = len(user_map)
num_movies = len(movie_map)

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
class Recommender(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embed  = nn.Embedding(num_users, 50)
        self.movie_embed = nn.Embedding(num_movies, 50)
        self.fc = nn.Sequential(
            nn.Linear(100, 128), nn.ReLU(), nn.Linear(128, 1)
        )
    def forward(self, user, movie):
        u = self.user_embed(user)
        m = self.movie_embed(movie)
        return self.fc(torch.cat([u, m], dim=1)).squeeze()

model = Recommender()

# ── Load saved weights OR train ──────────
if os.path.exists("model.pth"):
    try:
        model.load_state_dict(torch.load("model.pth", map_location="cpu"))
        model.eval()
    except:
        # Saved model incompatible — retrain
        _retrain = True
    else:
        _retrain = False
else:
    _retrain = True

if _retrain:
    users_t   = torch.tensor(data["user"].values,   dtype=torch.long)
    movies_t  = torch.tensor(data["movie"].values,  dtype=torch.long)
    ratings_t = torch.tensor(data["rating"].values, dtype=torch.float32)
    loss_fn   = nn.MSELoss()
    opt       = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(3):
        preds = model(users_t, movies_t)
        loss  = loss_fn(preds, ratings_t)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.save(model.state_dict(), "model.pth")
    model.eval()

with open("user_map.pkl",  "wb") as f: pickle.dump(user_map,  f)
with open("movie_map.pkl", "wb") as f: pickle.dump(movie_map, f)
movies.to_csv("movies_processed.csv", index=False)

# ─────────────────────────────────────────
# GENRE / TRAIT / EMOTION MAPS
# ─────────────────────────────────────────
GENRE_TRAIT_MAP = {
    "Sci-Fi":      "Curiosity",
    "Documentary": "Curiosity",
    "Mystery":     "Curiosity",
    "Adventure":   "Adventurous",
    "Action":      "Adventurous",
    "Western":     "Adventurous",
    "Drama":       "Empathetic",
    "Romance":     "Empathetic",
    "War":         "Empathetic",
    "Comedy":      "Light-hearted",
    "Animation":   "Light-hearted",
    "Children":    "Light-hearted",
    "Horror":      "Thrill-seeker",
    "Thriller":    "Thrill-seeker",
    "Crime":       "Thrill-seeker",
    "Film-Noir":   "Thrill-seeker",
    "Musical":     "Creative",
    "Fantasy":     "Creative",
    "IMAX":        "Adventurous",
}

TRAIT_EMOJI = {
    "Curiosity":     "🔭",
    "Adventurous":   "🏔️",
    "Empathetic":    "💙",
    "Light-hearted": "😄",
    "Thrill-seeker": "⚡",
    "Creative":      "🎨",
}

GENRE_EMOTION_MAP = {
    "Action": "Exciting", "Adventure": "Exciting",
    "Thriller": "Exciting", "Horror": "Exciting",
    "Fantasy": "Exciting", "Western": "Exciting",
    "Sci-Fi": "Thought-provoking", "Mystery": "Thought-provoking",
    "Documentary": "Thought-provoking", "Drama": "Thought-provoking",
    "War": "Thought-provoking", "Film-Noir": "Thought-provoking",
    "Crime": "Thought-provoking",
    "Comedy": "Light-hearted", "Animation": "Light-hearted",
    "Children": "Light-hearted", "Musical": "Light-hearted",
    "Romance": "Light-hearted",
}

EMOTION_EMOJI = {
    "Exciting":          "⚡",
    "Thought-provoking": "🧠",
    "Light-hearted":     "☀️",
}

# ─────────────────────────────────────────
# USER PREFERENCES
# ─────────────────────────────────────────
def get_user_preferences(user_id):
    user_ratings = data[data["userId"] == user_id].copy()
    if os.path.exists("user_data.csv"):
        ud = pd.read_csv("user_data.csv")
        ud["userId"] = ud["userId"].astype(str)
        nr = ud[ud["userId"] == str(user_id)][["movieId","rating"]].copy()
        if not nr.empty:
            user_ratings = pd.concat([user_ratings, nr], ignore_index=True)
    if user_ratings.empty:
        return {}
    merged = user_ratings.merge(movies, on="movieId", how="left")
    genre_scores = {}
    for _, row in merged.iterrows():
        for g in str(row.get("genres", "")).split("|"):
            g = g.strip()
            if g:
                genre_scores.setdefault(g, []).append(float(row["rating"]))
    return {g: sum(v)/len(v) for g, v in genre_scores.items()}

# ─────────────────────────────────────────
# PERSONALITY TRAITS  (fixed: relative, not absolute threshold)
# ─────────────────────────────────────────
def get_personality_traits(user_id):
    prefs = get_user_preferences(user_id)
    if not prefs:
        return []

    # Use relative threshold: above the user's own average
    user_mean = sum(prefs.values()) / len(prefs)
    # Lower bound so even if all ratings are low, we still show something
    threshold = max(user_mean * 0.85, 0.5)

    trait_scores = {}
    for genre, avg_rating in prefs.items():
        if avg_rating < threshold:
            continue
        trait = GENRE_TRAIT_MAP.get(genre)
        if trait:
            trait_scores.setdefault(trait, []).append(avg_rating)

    if not trait_scores:
        # Fallback: just take whatever genres they rated highest
        top_genres = sorted(prefs.items(), key=lambda x: -x[1])[:5]
        for genre, score in top_genres:
            trait = GENRE_TRAIT_MAP.get(genre)
            if trait:
                trait_scores.setdefault(trait, []).append(score)

    averaged = {t: sum(v)/len(v) for t, v in trait_scores.items()}
    strong = sorted(averaged.items(), key=lambda x: -x[1])
    return strong[:3]

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_popular_movies(top_k=10):
    try:
        popular = (
            data.groupby("movieId")["rating"]
            .mean().sort_values(ascending=False).head(100)
        )
        sample = popular.sample(top_k)
        return movies[movies["movieId"].isin(sample.index)][["title"]]
    except:
        return movies.sample(top_k)[["title"]]

def get_emotional_tag(genres_str):
    genres = [g.strip() for g in str(genres_str).split("|")]
    counts = {}
    for g in genres:
        e = GENRE_EMOTION_MAP.get(g)
        if e:
            counts[e] = counts.get(e, 0) + 1
    return max(counts, key=counts.get) if counts else "Thought-provoking"

def compute_confidence_risk(pred_score, user_prefs, genres_str):
    clamped    = max(0.5, min(5.0, float(pred_score)))
    confidence = int(round((clamped - 0.5) / 4.5 * 100))
    confidence = max(10, min(99, confidence))
    if not user_prefs:
        return confidence, "Medium"
    genres     = [g.strip() for g in str(genres_str).split("|")]
    genre_avgs = [user_prefs[g] for g in genres if g in user_prefs]
    if not genre_avgs:
        return confidence, "High"
    avg_pref   = sum(genre_avgs) / len(genre_avgs)
    user_mean  = sum(user_prefs.values()) / len(user_prefs)
    deviation  = abs(avg_pref - user_mean)
    risk = "Low" if deviation < 0.5 else ("Medium" if deviation < 1.2 else "High")
    return confidence, risk

def get_recommendation_reason(genres_str, user_prefs, personality_traits):
    genres  = [g.strip() for g in str(genres_str).split("|")]
    reasons = []
    if user_prefs:
        user_mean   = sum(user_prefs.values()) / len(user_prefs)
        top_matches = [g for g in genres if g in user_prefs and user_prefs[g] >= user_mean]
        if top_matches:
            reasons.append(f"Matches your love of {top_matches[0]}")
    if personality_traits:
        top_trait   = personality_traits[0][0]
        trait_genres = [g for g, t in GENRE_TRAIT_MAP.items() if t == top_trait]
        if any(g in trait_genres for g in genres):
            emoji = TRAIT_EMOJI.get(top_trait, "")
            reasons.append(f"Fits your {emoji} {top_trait} side")
    if not reasons:
        reasons.append("Highly rated by similar users")
    return " · ".join(reasons[:2])

def get_taste_evolution(user_id):
    rows = []
    orig = data[data["userId"] == user_id][["movieId","rating"]].copy()
    orig["source"] = "orig"
    rows.append(orig)
    if os.path.exists("user_data.csv"):
        ud = pd.read_csv("user_data.csv")
        ud["userId"] = ud["userId"].astype(str)
        ur = ud[ud["userId"] == str(user_id)][["movieId","rating"]].copy()
        if not ur.empty:
            ur["source"] = "new"
            rows.append(ur)
    all_r = pd.concat(rows, ignore_index=True)
    if len(all_r) < 4:
        return None, None
    mid    = len(all_r) // 2
    early  = all_r.iloc[:mid].merge(movies, on="movieId", how="left")
    recent = all_r.iloc[mid:].merge(movies, on="movieId", how="left")
    def top_genre(df):
        gs = {}
        for _, row in df.iterrows():
            for g in str(row.get("genres","")).split("|"):
                g = g.strip()
                if g:
                    gs.setdefault(g, []).append(float(row["rating"]))
        if not gs:
            return None
        return max(gs, key=lambda g: sum(gs[g])/len(gs[g]))
    return top_genre(early), top_genre(recent)

def surprise_me(user_id, top_k=10):
    prefs          = get_user_preferences(user_id)
    comfort_genres = set(g for g, s in prefs.items() if s >= max(list(prefs.values()) or [0]) * 0.8) if prefs else set()
    def is_outside(genres_str):
        gs = set(g.strip() for g in str(genres_str).split("|"))
        return len(gs & comfort_genres) == 0
    candidates = movies[movies["genres"].apply(is_outside)]
    if len(candidates) < top_k:
        candidates = movies
    pop = data.groupby("movieId")["rating"].mean()
    candidates = candidates.copy()
    candidates["pop_score"] = candidates["movieId"].map(pop).fillna(2.5)
    candidates = candidates.sort_values("pop_score", ascending=False).head(50)
    return candidates.sample(min(top_k, len(candidates)))[["title"]]

def one_perfect_recommendation(user_id):
    prefs      = get_user_preferences(user_id)
    all_movies = list(movie_map.keys())
    if user_id in user_map:
        user_idx      = user_map[user_id]
        movies_tensor = torch.tensor([movie_map[m] for m in all_movies])
        users_tensor  = torch.tensor([user_idx] * len(all_movies))
        with torch.no_grad():
            preds = model(users_tensor, movies_tensor).numpy()
    else:
        preds = np.random.uniform(0.01, 0.1, size=len(all_movies))
    scores = {}
    for i, mid in enumerate(all_movies):
        row = movies[movies["movieId"] == mid]
        if row.empty:
            continue
        genres = str(row.iloc[0]["genres"]).split("|")
        boost  = sum(prefs.get(g.strip(), 0) * 0.2 for g in genres)
        scores[mid] = float(preds[i]) + boost
    best_id  = max(scores, key=scores.get)
    best_row = movies[movies["movieId"] == best_id]
    conf, risk = compute_confidence_risk(
        scores[best_id], prefs,
        best_row.iloc[0]["genres"] if not best_row.empty else ""
    )
    return best_row[["title"]], conf, risk

def recommend_movies_with_personalization(user_id, top_k=10):
    return _recommend(user_id, top_k)["df"]

def recommend_with_metadata(user_id, top_k=10):
    return _recommend(user_id, top_k)

def _recommend(user_id, top_k=10):
    prefs      = get_user_preferences(user_id)
    traits     = get_personality_traits(user_id)
    all_movies = list(movie_map.keys())
    if user_id in user_map:
        user_idx      = user_map[user_id]
        movies_tensor = torch.tensor([movie_map[m] for m in all_movies])
        users_tensor  = torch.tensor([user_idx] * len(all_movies))
        with torch.no_grad():
            predictions = model(users_tensor, movies_tensor).numpy()
    else:
        if not prefs:
            df = get_popular_movies(top_k)
            return {"df": df, "meta": {}, "traits": []}
        predictions = np.random.uniform(0.01, 0.1, size=len(all_movies))
    boost_w = 0.5 if user_id not in user_map else 0.1
    for i, mid in enumerate(all_movies):
        row = movies[movies["movieId"] == mid]
        if row.empty:
            continue
        for g in str(row.iloc[0]["genres"]).split("|"):
            if g.strip() in prefs:
                predictions[i] += prefs[g.strip()] * boost_w
    top_idx    = predictions.argsort()[-(top_k * 3):][::-1]
    rec_ids    = [all_movies[i] for i in top_idx]
    result     = movies[movies["movieId"].isin(rec_ids)][["movieId","title","genres"]].head(top_k * 2)
    id_to_pred = {all_movies[i]: float(predictions[i]) for i in range(len(all_movies))}
    meta = {}
    for _, row in result.iterrows():
        conf, risk = compute_confidence_risk(id_to_pred.get(row["movieId"], 3.0), prefs, row["genres"])
        meta[row["title"]] = {
            "confidence": conf,
            "risk":       risk,
            "emotion":    get_emotional_tag(row["genres"]),
            "reason":     get_recommendation_reason(row["genres"], prefs, traits),
            "global_avg": round(global_avg_ratings.get(row["movieId"], 3.0), 1),
        }
    return {"df": result[["title"]].head(top_k), "meta": meta, "traits": traits}
