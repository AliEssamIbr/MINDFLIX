import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
ratings = pd.read_csv("data/ratings.csv")
movies  = pd.read_csv("data/movies.csv")
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

# Fast genre lookup — avoids repeated DataFrame merges that caused KeyError:'genres'
_movie_genres = movies.set_index("movieId")["genres"].to_dict()

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
class Recommender(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embed  = nn.Embedding(num_users, 50)
        self.movie_embed = nn.Embedding(num_movies, 50)
        self.fc = nn.Sequential(nn.Linear(100, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, user, movie):
        return self.fc(torch.cat([self.user_embed(user), self.movie_embed(movie)], dim=1)).squeeze()

model = Recommender()
if os.path.exists("model.pth"):
    try:
        model.load_state_dict(torch.load("model.pth", map_location="cpu"))
        model.eval()
        _retrain = False
    except:
        _retrain = True
else:
    _retrain = True

if _retrain:
    ut = torch.tensor(data["user"].values,   dtype=torch.long)
    mt = torch.tensor(data["movie"].values,  dtype=torch.long)
    rt = torch.tensor(data["rating"].values, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(3):
        loss = nn.MSELoss()(model(ut, mt), rt)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.save(model.state_dict(), "model.pth")
    model.eval()

with open("user_map.pkl",  "wb") as f: pickle.dump(user_map,  f)
with open("movie_map.pkl", "wb") as f: pickle.dump(movie_map, f)
movies.to_csv("movies_processed.csv", index=False)

# ─────────────────────────────────────────
# MAPS
# ─────────────────────────────────────────
GENRE_TRAIT_MAP = {
    "Sci-Fi":"Curious",      "Documentary":"Curious",    "Mystery":"Curious",
    "Adventure":"Adventurous","Action":"Adventurous",    "Western":"Adventurous","IMAX":"Adventurous",
    "Drama":"Empathetic",    "War":"Empathetic",
    "Romance":"Romantic",
    "Comedy":"Comedy Lover", "Animation":"Comedy Lover", "Children":"Comedy Lover",
    "Horror":"Thrill-seeker","Thriller":"Thrill-seeker", "Crime":"Thrill-seeker","Film-Noir":"Thrill-seeker",
    "Musical":"Creative",    "Fantasy":"Creative",
}

TRAIT_EMOJI = {
    "Curious":"🔭", "Adventurous":"🏔️", "Empathetic":"💙",
    "Romantic":"💕", "Comedy Lover":"😄", "Thrill-seeker":"⚡", "Creative":"🎨",
}

TRAIT_DESC = {
    "Curious":       "You love films that challenge your mind and explore new ideas.",
    "Adventurous":   "You're drawn to action, journeys, and the thrill of the unknown.",
    "Empathetic":    "You connect deeply with human stories and emotional narratives.",
    "Romantic":      "You appreciate love stories and deep human connections.",
    "Comedy Lover":  "You appreciate humor, warmth, and feel-good entertainment.",
    "Thrill-seeker": "You crave suspense, danger, and edge-of-your-seat moments.",
    "Creative":      "You're captivated by imagination, fantasy, and artistic expression.",
}

GENRE_EMOTION_MAP = {
    "Action":"Exciting","Adventure":"Exciting","Thriller":"Exciting",
    "Horror":"Exciting","Fantasy":"Exciting","Western":"Exciting",
    "Sci-Fi":"Thought-provoking","Mystery":"Thought-provoking","Documentary":"Thought-provoking",
    "Drama":"Thought-provoking","War":"Thought-provoking","Film-Noir":"Thought-provoking","Crime":"Thought-provoking",
    "Comedy":"Light-hearted","Animation":"Light-hearted","Children":"Light-hearted",
    "Musical":"Light-hearted","Romance":"Light-hearted",
}

EMOTION_EMOJI = {"Exciting":"⚡","Thought-provoking":"🧠","Light-hearted":"☀️"}

# ─────────────────────────────────────────
# CORE HELPER — loads user ratings safely
#
# THE BUG THIS FIXES:
#   Old code did: concat(original_rows_with_genres, new_rows_without_genres)
#   then: merged = all.merge(movies, on="movieId")
#   When the user is new (0 rows in original data), concat produced a df
#   where genres column existed but was all NaN. The subsequent merge then
#   had a column conflict → KeyError: 'genres' → swallowed silently → [] returned.
#
# THE FIX:
#   Collect ONLY movieId+rating from both sources, then map genres once
#   using a pre-built dictionary. No merge, no column conflicts, always works.
# ─────────────────────────────────────────
def _load_user_ratings(user_id):
    """Return DataFrame[movieId, rating, genres] for a user. Always safe."""
    orig = data[data["userId"] == user_id][["movieId", "rating"]].copy()

    new_rows = pd.DataFrame(columns=["movieId", "rating"])
    if os.path.exists("user_data.csv"):
        ud = pd.read_csv("user_data.csv")
        ud["userId"] = ud["userId"].astype(str)
        nr = ud[ud["userId"] == str(user_id)][["movieId", "rating"]].copy()
        if not nr.empty:
            new_rows = nr

    combined = pd.concat([orig, new_rows], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=["movieId", "rating", "genres"])

    combined["genres"] = combined["movieId"].map(_movie_genres).fillna("")
    return combined

# ─────────────────────────────────────────
# USER PREFERENCES
# ─────────────────────────────────────────
def get_user_preferences(user_id):
    df = _load_user_ratings(user_id)
    if df.empty:
        return {}
    gs = {}
    for _, row in df.iterrows():
        for g in str(row["genres"]).split("|"):
            g = g.strip()
            if g and g != "(no genres listed)":
                gs.setdefault(g, []).append(float(row["rating"]))
    return {g: sum(v) / len(v) for g, v in gs.items()}

# ─────────────────────────────────────────
# PERSONALITY TRAITS
# Score = 60% genre frequency + 40% avg rating.
# Frequency dominates so traits appear even if all ratings are low.
#
# SECOND BUG FIXED HERE:
#   Old return line used 'data' as the loop variable name, shadowing
#   the global DataFrame. Renamed to 'd' to avoid the conflict.
# ─────────────────────────────────────────
def get_personality_traits(user_id):
    df = _load_user_ratings(user_id)
    if df.empty:
        return []

    genre_counts = {}
    genre_scores = {}
    for _, row in df.iterrows():
        for g in str(row["genres"]).split("|"):
            g = g.strip()
            if g and g != "(no genres listed)":
                genre_counts[g] = genre_counts.get(g, 0) + 1
                genre_scores.setdefault(g, []).append(float(row["rating"]))

    if not genre_counts:
        return []

    max_count = max(genre_counts.values())
    trait_scores = {}
    for genre, count in genre_counts.items():
        trait = GENRE_TRAIT_MAP.get(genre)
        if not trait:
            continue
        freq   = count / max_count
        avg_r  = sum(genre_scores[genre]) / len(genre_scores[genre])
        norm_r = (avg_r - 0.5) / 4.5
        score  = freq * 0.6 + norm_r * 0.4
        if trait not in trait_scores or score > trait_scores[trait][0]:
            trait_scores[trait] = (score, avg_r, count)

    if not trait_scores:
        return []

    sorted_traits = sorted(trait_scores.items(), key=lambda x: -x[1][0])
    # 'd' instead of 'data' to avoid shadowing the global DataFrame
    return [(trait, round(d[2], 0), round(d[1], 1)) for trait, d in sorted_traits[:4]]

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def get_popular_movies(top_k=10):
    try:
        popular = data.groupby("movieId")["rating"].mean().sort_values(ascending=False).head(100)
        return movies[movies["movieId"].isin(popular.sample(top_k).index)][["title"]]
    except:
        return movies.sample(top_k)[["title"]]

def get_emotional_tag(genres_str):
    counts = {}
    for g in str(genres_str).split("|"):
        e = GENRE_EMOTION_MAP.get(g.strip())
        if e: counts[e] = counts.get(e, 0) + 1
    return max(counts, key=counts.get) if counts else "Thought-provoking"

def compute_confidence_risk(pred_score, user_prefs, genres_str):
    clamped    = max(0.5, min(5.0, float(pred_score)))
    confidence = max(10, min(99, int(round((clamped - 0.5) / 4.5 * 100))))
    if not user_prefs:
        return confidence, "Medium"
    genres     = [g.strip() for g in str(genres_str).split("|")]
    genre_avgs = [user_prefs[g] for g in genres if g in user_prefs]
    if not genre_avgs:
        return confidence, "High"
    deviation  = abs(sum(genre_avgs) / len(genre_avgs) - sum(user_prefs.values()) / len(user_prefs))
    return confidence, ("Low" if deviation < 0.5 else "Medium" if deviation < 1.2 else "High")

def get_recommendation_reason(genres_str, user_prefs, personality_traits):
    genres  = [g.strip() for g in str(genres_str).split("|")]
    reasons = []
    if user_prefs:
        um  = sum(user_prefs.values()) / len(user_prefs)
        top = [g for g in genres if g in user_prefs and user_prefs[g] >= um]
        if top: reasons.append(f"Matches your love of {top[0]}")
    if personality_traits:
        top_trait    = personality_traits[0][0]
        trait_genres = [g for g, t in GENRE_TRAIT_MAP.items() if t == top_trait]
        if any(g in trait_genres for g in genres):
            reasons.append(f"Fits your {TRAIT_EMOJI.get(top_trait, '')} {top_trait} side")
    if not reasons:
        reasons.append("Highly rated by similar users")
    return " · ".join(reasons[:2])

def get_taste_evolution(user_id):
    df = _load_user_ratings(user_id)
    if len(df) < 4:
        return None, None
    mid = len(df) // 2

    def top_genre(subset):
        gs = {}
        for _, row in subset.iterrows():
            for g in str(row["genres"]).split("|"):
                g = g.strip()
                if g and g != "(no genres listed)":
                    gs.setdefault(g, []).append(float(row["rating"]))
        return max(gs, key=lambda g: len(gs[g])) if gs else None

    return top_genre(df.iloc[:mid]), top_genre(df.iloc[mid:])

def surprise_me(user_id, top_k=10):
    prefs   = get_user_preferences(user_id)
    comfort = set(g for g, s in prefs.items()
                  if s >= max(list(prefs.values()) or [0]) * 0.8) if prefs else set()
    cands   = movies[movies["genres"].apply(
        lambda g: not any(x.strip() in comfort for x in str(g).split("|")))]
    if len(cands) < top_k:
        cands = movies
    pop   = data.groupby("movieId")["rating"].mean()
    cands = cands.copy()
    cands["pop"] = cands["movieId"].map(pop).fillna(2.5)
    return cands.nlargest(50, "pop").sample(min(top_k, len(cands)))[["title"]]

def one_perfect_recommendation(user_id):
    prefs = get_user_preferences(user_id)
    all_m = list(movie_map.keys())
    if user_id in user_map:
        with torch.no_grad():
            preds = model(
                torch.tensor([user_map[user_id]] * len(all_m)),
                torch.tensor([movie_map[m] for m in all_m])
            ).numpy()
    else:
        preds = np.random.uniform(0.01, 0.1, size=len(all_m))
    scores = {}
    for i, mid in enumerate(all_m):
        genres_str = _movie_genres.get(mid, "")
        boost = sum(prefs.get(g.strip(), 0) * 0.2 for g in genres_str.split("|"))
        scores[mid] = float(preds[i]) + boost
    best_id  = max(scores, key=scores.get)
    best_row = movies[movies["movieId"] == best_id]
    conf, risk = compute_confidence_risk(
        scores[best_id], prefs,
        best_row.iloc[0]["genres"] if not best_row.empty else "")
    return best_row[["title"]], conf, risk

def recommend_movies_with_personalization(user_id, top_k=10):
    return _recommend(user_id, top_k)["df"]

def recommend_with_metadata(user_id, top_k=10):
    return _recommend(user_id, top_k)

def _recommend(user_id, top_k=10):
    prefs  = get_user_preferences(user_id)
    traits = get_personality_traits(user_id)
    all_m  = list(movie_map.keys())

    if user_id in user_map:
        with torch.no_grad():
            predictions = model(
                torch.tensor([user_map[user_id]] * len(all_m)),
                torch.tensor([movie_map[m] for m in all_m])
            ).numpy()
    else:
        if not prefs:
            return {"df": get_popular_movies(top_k), "meta": {}, "traits": []}
        predictions = np.random.uniform(0.01, 0.1, size=len(all_m))

    bw = 0.5 if user_id not in user_map else 0.1
    for i, mid in enumerate(all_m):
        for g in str(_movie_genres.get(mid, "")).split("|"):
            if g.strip() in prefs:
                predictions[i] += prefs[g.strip()] * bw

    rec_ids = [all_m[i] for i in predictions.argsort()[-(top_k * 3):][::-1]]
    result  = movies[movies["movieId"].isin(rec_ids)][["movieId", "title", "genres"]].head(top_k * 2)
    id2pred = {all_m[i]: float(predictions[i]) for i in range(len(all_m))}

    meta = {}
    for _, row in result.iterrows():
        conf, risk = compute_confidence_risk(id2pred.get(row["movieId"], 3.0), prefs, row["genres"])
        meta[row["title"]] = {
            "confidence": conf,
            "risk":       risk,
            "emotion":    get_emotional_tag(row["genres"]),
            "reason":     get_recommendation_reason(row["genres"], prefs, traits),
            "global_avg": round(global_avg_ratings.get(row["movieId"], 3.0), 1),
        }
    return {"df": result[["title"]].head(top_k), "meta": meta, "traits": traits}
