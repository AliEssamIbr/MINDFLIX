import streamlit as st
import requests
from PIL import Image
import pandas as pd
import os, csv, re, random
import numpy as np

from recommender import (
    recommend_with_metadata, get_popular_movies,
    get_personality_traits, get_taste_evolution,
    get_user_preferences, surprise_me,
    one_perfect_recommendation, global_avg_ratings,
    TRAIT_EMOJI, EMOTION_EMOJI,
)

API_KEY = "YOUR API KEY HERE" # API KEY FROM www.themoviedb.org
st.set_page_config(page_title="AI Movie Recommender", layout="wide")

ALL_GENRES = [
    "Action","Adventure","Animation","Children","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","IMAX",
    "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"
]
RISK_COLOR    = {"Low": "#48bb78", "Medium": "#ecc94b", "High": "#fc8181"}
EMOTION_COLOR = {
    "Exciting":          "rgba(252,129,74,0.18)",
    "Thought-provoking": "rgba(99,179,237,0.18)",
    "Light-hearted":     "rgba(104,211,145,0.18)",
}

# ─────────────────────────────────────────
# CSS  — geometric SVG background + UI
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Geometric background ── */
.stApp {
    background-color: #0a0a14;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cdefs%3E%3Cpattern id='g' width='120' height='120' patternUnits='userSpaceOnUse'%3E%3Cpolygon points='60,5 115,35 115,85 60,115 5,85 5,35' fill='none' stroke='%231a1a3a' stroke-width='1'/%3E%3Cline x1='60' y1='5' x2='60' y2='115' stroke='%2316163a' stroke-width='0.5'/%3E%3Cline x1='5' y1='35' x2='115' y2='85' stroke='%2316163a' stroke-width='0.5'/%3E%3Cline x1='115' y1='35' x2='5' y2='85' stroke='%2316163a' stroke-width='0.5'/%3E%3C/pattern%3E%3C/defs%3E%3Crect width='120' height='120' fill='url(%23g)'/%3E%3C/svg%3E");
    color: #f0f0f0;
}

/* glassmorphism panels */
.glass {
    background: rgba(15,15,35,0.82);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.07);
}

/* ── section header ── */
.section-header {
    font-size: 20px; font-weight: 700; color: #e8e8f0;
    margin: 28px 0 14px 0; padding-bottom: 8px;
    border-bottom: 2px solid rgba(99,179,237,0.35);
}

/* ── genre pill ── */
.genre-tag {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    color: #c8c8e0;
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 20px;
    padding: 2px 10px; margin: 2px;
    font-size: 11px; font-weight: 500;
}

/* ── personality trait pill ── */
.trait-pill {
    display: inline-block;
    background: rgba(159,122,234,0.18);
    color: #c4a8ff;
    border: 1px solid rgba(159,122,234,0.28);
    border-radius: 20px;
    padding: 5px 14px; margin: 3px;
    font-size: 13px; font-weight: 600;
}

/* ── movie detail panel ── */
.movie-detail-panel {
    background: rgba(20,20,50,0.9);
    border-radius: 12px; padding: 20px;
    margin: 10px 0 20px 0;
    border: 1px solid rgba(99,179,237,0.25);
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}

/* ── rated row ── */
.rated-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 13px; color: #bbb;
}
.rated-score {
    background: rgba(99,179,237,0.18); color: #63b3ed;
    border-radius: 20px; padding: 2px 10px;
    font-weight: 600; font-size: 12px; white-space: nowrap;
}

/* ── confidence bar ── */
.conf-wrap {
    background: rgba(255,255,255,0.07);
    border-radius: 20px; height: 7px; width: 100%; margin: 5px 0;
}
.conf-fill {
    height: 7px; border-radius: 20px;
    background: linear-gradient(90deg,#63b3ed,#9f7aea);
}

/* ── meta badge ── */
.meta-badge {
    display: inline-block; border-radius: 20px;
    padding: 3px 12px; font-size: 12px; font-weight: 600; margin: 2px 3px;
}

/* ── one perfect card ── */
.one-perfect-card {
    background: linear-gradient(135deg,rgba(20,20,60,0.95) 0%,rgba(40,20,70,0.95) 100%);
    border: 2px solid rgba(159,122,234,0.45);
    border-radius: 16px; padding: 24px;
    box-shadow: 0 0 40px rgba(159,122,234,0.15);
}

/* ── profile card ── */
.profile-card {
    background: rgba(255,255,255,0.04);
    border-radius: 12px; padding: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 14px;
}

/* ── no poster placeholder ── */
.no-poster {
    height: 200px; background: rgba(30,30,60,0.7);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 38px; border: 1px solid rgba(255,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CACHED DATA HELPERS  (no rerun on click)
# ─────────────────────────────────────────
movies_df = pd.read_csv("movies_processed.csv")

@st.cache_data(show_spinner=False)
def cached_poster(title):
    file_path = f"posters/{re.sub(r'[\\/*?:\"<>|]', '', title).replace(' ','_')}.jpg"
    if os.path.exists(file_path):
        return file_path
    try:
        d = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"api_key": API_KEY, "query": re.sub(r"\(\d{4}\)","",title).strip()}
        ).json()
        if d.get("results"):
            pp = d["results"][0].get("poster_path")
            if pp:
                img = requests.get(f"https://image.tmdb.org/t/p/w500{pp}").content
                with open(file_path, "wb") as f: f.write(img)
                return file_path
    except: pass
    return None

@st.cache_data(show_spinner=False)
def cached_genres(title):
    row = movies_df[movies_df["title"] == title]
    if row.empty: return []
    g = row.iloc[0].get("genres","")
    if pd.isna(g) or not g: return []
    return [x.strip() for x in str(g).split("|") if x.strip() and x.strip() != "(no genres listed)"]

@st.cache_data(ttl=30, show_spinner=False)
def cached_recs(user_id, top_k, seed):
    random.seed(seed)
    bundle = recommend_with_metadata(user_id, top_k=top_k + 15)
    titles = list(bundle["df"]["title"])
    random.shuffle(titles)
    chosen = titles[:top_k]
    df = bundle["df"][bundle["df"]["title"].isin(chosen)]
    meta = {t: bundle["meta"][t] for t in chosen if t in bundle["meta"]}
    return df, meta, bundle.get("traits", [])

@st.cache_data(ttl=30, show_spinner=False)
def cached_popular(top_k, seed):
    np.random.seed(seed)
    return get_popular_movies(top_k)

@st.cache_data(ttl=30, show_spinner=False)
def cached_surprise(user_id, top_k, seed):
    np.random.seed(seed)
    return surprise_me(user_id, top_k)

@st.cache_data(ttl=60, show_spinner=False)
def cached_perfect(user_id):
    return one_perfect_recommendation(user_id)

@st.cache_data(ttl=60, show_spinner=False)
def cached_traits(user_id):
    return get_personality_traits(user_id)

@st.cache_data(ttl=60, show_spinner=False)
def cached_evolution(user_id):
    return get_taste_evolution(user_id)

@st.cache_data(ttl=30, show_spinner=False)
def cached_user_ratings(user_id):
    if not os.path.exists("user_data.csv"): return []
    df = pd.read_csv("user_data.csv")
    df["userId"] = df["userId"].astype(str)
    ur = df[df["userId"] == str(user_id)].merge(
        movies_df[["movieId","title"]], on="movieId", how="left"
    ).dropna(subset=["title"])
    return list(zip(ur["title"], ur["rating"]))[::-1]

def invalidate_user_cache(user_id):
    cached_recs.clear()
    cached_popular.clear()
    cached_perfect.clear()
    cached_traits.clear()
    cached_evolution.clear()
    cached_user_ratings.clear()

# ─────────────────────────────────────────
# SMALL HELPERS
# ─────────────────────────────────────────
def clean_title(t): return re.sub(r"\(\d{4}\)","",t).strip()

def render_genre_tags(genres):
    if not genres: return ""
    return '<div style="padding:4px 0 6px 0;">' + "".join(
        f'<span class="genre-tag">{g}</span>' for g in genres
    ) + "</div>"

def render_meta_badges(m):
    if not m: return ""
    conf = m.get("confidence",0); risk = m.get("risk","Medium"); emotion = m.get("emotion","")
    rc = RISK_COLOR.get(risk,"#ecc94b"); ec = EMOTION_COLOR.get(emotion,"rgba(255,255,255,0.1)")
    ee = EMOTION_EMOJI.get(emotion,"")
    return f"""
    <div style='margin:6px 0 4px 0;'>
      <div style='display:flex;justify-content:space-between;font-size:11px;color:#aaa;margin-bottom:3px;'>
        <span>Confidence</span><span style='color:#63b3ed;font-weight:700;'>{conf}%</span>
      </div>
      <div class='conf-wrap'><div class='conf-fill' style='width:{conf}%;'></div></div>
    </div>
    <span class='meta-badge' style='background:{ec};color:#f0f0f0;'>{ee} {emotion}</span>
    <span class='meta-badge' style='background:{rc}22;color:{rc};border:1px solid {rc}44;'>Risk: {risk}</span>
    """

def save_rating(user_id, title, rating):
    row = movies_df[movies_df["title"] == title]
    if row.empty: return False
    mid = row.iloc[0]["movieId"]
    exists = os.path.isfile("user_data.csv")
    with open("user_data.csv","a",newline="") as f:
        w = csv.writer(f)
        if not exists: w.writerow(["userId","movieId","rating"])
        w.writerow([user_id, mid, rating])
    return True

def star_label(r):
    return "⭐"*int(r) + ("½" if r%1>=0.5 else "")

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
for k,v in {
    "user_id":None,"logged_in":False,
    "expanded_movie":None,"pending_rating":{},"rating_submitted":set(),
    "rec_seed":0,"pop_seed":0,"show_profile":False,
    "search_expanded":None,"surprise_mode":False,
}.items():
    if k not in st.session_state: st.session_state[k]=v

# ─────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────
if not st.session_state.logged_in:
    st.markdown("""
    <div style='text-align:center;padding:60px 20px 20px;'>
        <div style='font-size:60px;margin-bottom:14px;'>🎬</div>
        <h1 style='color:#e8e8ff;font-size:38px;margin-bottom:6px;letter-spacing:-1px;'>AI Movie Recommender</h1>
        <p style='color:#888;font-size:15px;margin-bottom:44px;'>
            Deep Learning · Explainable AI · Personality Profiling
        </p>
    </div>""", unsafe_allow_html=True)
    _, col, _ = st.columns([1,1.3,1])
    with col:
        with st.container():
            st.markdown("""
            <div class='glass' style='padding:32px 28px;'>
            <h3 style='color:#e0e0ff;text-align:center;margin-bottom:20px;font-size:18px;'>
                Choose your profile
            </h3></div>""", unsafe_allow_html=True)
            uid_in = st.number_input("uid", min_value=1, step=1, label_visibility="collapsed")
            if st.button("▶  Log In", use_container_width=True):
                st.session_state.user_id   = int(uid_in)
                st.session_state.logged_in = True
                st.rerun()
            st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:18px 0;'>", unsafe_allow_html=True)
            st.markdown("<p style='color:#888;text-align:center;font-size:12px;'>New here?</p>", unsafe_allow_html=True)
            if st.button("🎲  Create New User", use_container_width=True):
                nid = 1
                if os.path.exists("user_data.csv"):
                    df = pd.read_csv("user_data.csv")
                    if not df.empty: nid = int(df["userId"].max())+1
                st.session_state.user_id   = nid
                st.session_state.logged_in = True
                st.rerun()
    st.stop()

# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
uid = st.session_state.user_id

# ── TOP BAR ──────────────────────────────
L, R = st.columns([5,3])
with L:
    st.markdown("<h2 style='color:#e8e8ff;margin:0;letter-spacing:-0.5px;'>🎬 AI Movie Recommender</h2>",
                unsafe_allow_html=True)
with R:
    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("👤 Profile"):
            st.session_state.show_profile = not st.session_state.show_profile
    with c2:
        st.markdown(f"<p style='color:#888;padding-top:8px;text-align:center;font-size:13px;'>User {uid}</p>",
                    unsafe_allow_html=True)
    with c3:
        if st.button("🚪 Logout"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin-bottom:4px;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# PROFILE PANEL
# ─────────────────────────────────────────
if st.session_state.show_profile:
    rated   = cached_user_ratings(uid)
    traits  = cached_traits(uid)
    early_g, recent_g = cached_evolution(uid)
    prefs   = get_user_preferences(uid)

    st.markdown(f"""
    <div class='profile-card'>
        <h3 style='color:#e0e0ff;margin-bottom:4px;'>👤 User {uid}</h3>
        <p style='color:#888;font-size:13px;'>
            Ratings: <strong style='color:#63b3ed;'>{len(rated)}</strong>
        </p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎬 My Ratings", "🧠 Personality", "📈 Taste Evolution"])

    with tab1:
        if not rated:
            st.info("No ratings yet — click any movie title to rate it!")
        else:
            seen = {}
            for title, rating in rated:
                if title not in seen: seen[title] = rating

            rows_html = ""
            for title, rating in list(seen.items())[:40]:
                mid_row  = movies_df[movies_df["title"]==title]
                glob_avg = 3.0
                if not mid_row.empty:
                    glob_avg = round(global_avg_ratings.get(mid_row.iloc[0]["movieId"], 3.0),1)
                diff = round(float(rating) - glob_avg, 1)
                diff_str = ""
                if diff > 0.5:
                    diff_str = f"<span style='color:#48bb78;font-size:10px;'> +{diff} vs crowd</span>"
                elif diff < -0.5:
                    diff_str = f"<span style='color:#fc8181;font-size:10px;'> {diff} vs crowd</span>"
                short = title[:44]+"…" if len(title)>44 else title
                rows_html += f"<div class='rated-row'><span>{short}{diff_str}</span><span class='rated-score'>{star_label(rating)} {rating}</span></div>"
            st.markdown(rows_html, unsafe_allow_html=True)

            st.markdown("#### 🎭 Genre Preferences")
            gs = {}
            for title, rating in seen.items():
                for g in cached_genres(title):
                    gs.setdefault(g,[]).append(float(rating))
            for g, score in sorted({g:round(sum(v)/len(v),2) for g,v in gs.items()}.items(), key=lambda x:-x[1])[:8]:
                st.progress(score/5.0, text=f"{g}  —  {score} ⭐")

    with tab2:
        if not traits:
            st.info("Rate at least 3–5 movies to unlock your personality profile!")
        else:
            st.markdown("<h4 style='color:#e0e0ff;margin-bottom:12px;'>Your Cinematic Personality</h4>",
                        unsafe_allow_html=True)
            st.markdown(
                '<div style="margin-bottom:16px;">' +
                "".join(f"<span class='trait-pill'>{TRAIT_EMOJI.get(t,'🎬')} {t}</span>" for t,_ in traits) +
                "</div>", unsafe_allow_html=True)
            desc = {
                "Curiosity":     "You love films that challenge your mind and explore new ideas.",
                "Adventurous":   "You're drawn to action and journeys into the unknown.",
                "Empathetic":    "You connect deeply with human stories and emotional narratives.",
                "Light-hearted": "You appreciate humor, warmth, and feel-good entertainment.",
                "Thrill-seeker": "You crave suspense, danger, and edge-of-your-seat moments.",
                "Creative":      "You're captivated by imagination, fantasy, and artistic expression.",
            }
            for t, score in traits:
                st.markdown(f"""
                <div style='background:rgba(159,122,234,0.08);border-radius:10px;
                            padding:12px 16px;margin-bottom:8px;
                            border:1px solid rgba(159,122,234,0.18);'>
                    <strong style='color:#b794f4;'>{TRAIT_EMOJI.get(t,"")} {t}</strong>
                    <span style='color:#666;font-size:12px;margin-left:8px;'>avg {score:.1f}⭐</span>
                    <p style='color:#aaa;font-size:13px;margin:4px 0 0;'>{desc.get(t,"")}</p>
                </div>""", unsafe_allow_html=True)

    with tab3:
        if not early_g or not recent_g:
            st.info("Rate more movies over time to see how your taste evolves!")
        else:
            c_e, c_a, c_r = st.columns([2,1,2])
            with c_e:
                st.markdown(f"""<div style='background:rgba(99,179,237,0.08);border-radius:12px;
                    padding:20px;border:1px solid rgba(99,179,237,0.18);text-align:center;'>
                    <p style='color:#888;font-size:11px;margin-bottom:4px;letter-spacing:1px;'>EARLY TASTE</p>
                    <h3 style='color:#63b3ed;margin:0;'>{early_g}</h3></div>""", unsafe_allow_html=True)
            with c_a:
                st.markdown("<div style='font-size:28px;text-align:center;color:#63b3ed;padding:14px;'>→</div>",
                            unsafe_allow_html=True)
            with c_r:
                st.markdown(f"""<div style='background:rgba(159,122,234,0.08);border-radius:12px;
                    padding:20px;border:1px solid rgba(159,122,234,0.18);text-align:center;'>
                    <p style='color:#888;font-size:11px;margin-bottom:4px;letter-spacing:1px;'>NOW</p>
                    <h3 style='color:#b794f4;margin:0;'>{recent_g}</h3></div>""", unsafe_allow_html=True)
            msg = "Your taste stayed consistent — you know what you like! 🎯" if early_g==recent_g \
                  else f"Your taste evolved from {early_g} → {recent_g}"
            st.markdown(f"<p style='color:#888;text-align:center;margin-top:12px;font-size:13px;'>{msg}</p>",
                        unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);margin:20px 0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# DETAIL PANEL
# ─────────────────────────────────────────
def show_detail_panel(title, section_key, meta=None):
    genres  = cached_genres(title)
    year_m  = re.search(r"\((\d{4})\)", title)
    year    = year_m.group(1) if year_m else "N/A"
    clean   = clean_title(title)
    rkey    = f"rating_{section_key}_{title}"

    mid_row  = movies_df[movies_df["title"]==title]
    glob_avg = 3.0
    if not mid_row.empty:
        glob_avg = round(global_avg_ratings.get(mid_row.iloc[0]["movieId"], 3.0),1)

    st.markdown(f"""
    <div class="movie-detail-panel">
        <h3 style='color:#e8e8ff;margin-bottom:4px;'>{title}</h3>
        <p style='color:#888;font-size:13px;margin-bottom:8px;'>📅 {year}
           &nbsp;·&nbsp; 🌍 Global avg: <strong style='color:#aaa;'>{glob_avg} ⭐</strong>
        </p>
        {render_genre_tags(genres)}
        {render_meta_badges(meta)}
        {"<p style='color:#888;font-size:12px;margin:8px 0 0;'>💡 "+meta['reason']+"</p>" if meta and meta.get('reason') else ""}
    </div>""", unsafe_allow_html=True)

    prev = next((r for t,r in cached_user_ratings(uid) if t==title), None)
    default = float(prev) if prev else st.session_state.pending_rating.get(rkey, 3.0)
    new_r = st.slider(f"⭐ Rate **{clean}**", 0.5, 5.0, default, 0.5,
                      key=f"slider_{section_key}_{title}")
    st.session_state.pending_rating[rkey] = new_r

    if rkey in st.session_state.rating_submitted:
        diff = round(new_r - glob_avg, 1)
        st.success(f"✅ Rated **{clean}** — {new_r} ⭐")
        if diff > 0.5:
            st.markdown(f"<p style='color:#48bb78;font-size:13px;'>You rated this <b>{diff}★ above</b> the global average — divergent taste! 🎯</p>", unsafe_allow_html=True)
        elif diff < -0.5:
            st.markdown(f"<p style='color:#fc8181;font-size:13px;'>You rated this <b>{abs(diff)}★ below</b> the global average.</p>", unsafe_allow_html=True)
    else:
        if st.button("Submit Rating", key=f"submit_{section_key}_{title}"):
            if save_rating(uid, title, new_r):
                st.session_state.rating_submitted.add(rkey)
                invalidate_user_cache(uid)
                st.rerun()

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin-top:16px;'>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────
# MOVIE GRID
# ─────────────────────────────────────────
def show_grid(df, section_key, meta_dict=None):
    titles = list(df["title"])
    for row_start in range(0, len(titles), 5):
        row_t = titles[row_start:row_start+5]
        cols  = st.columns(5)
        for ci, title in enumerate(row_t):
            with cols[ci]:
                m      = (meta_dict or {}).get(title, {})
                genres = cached_genres(title)
                poster = cached_poster(title)

                if poster and os.path.exists(poster):
                    try: st.image(Image.open(poster), use_container_width=True)
                    except: st.markdown("<div class='no-poster'>🎞️</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='no-poster'>🎞️</div>", unsafe_allow_html=True)

                lbl = title[:28]+"…" if len(title)>28 else title
                if st.button(lbl, key=f"btn_{section_key}_{title}", use_container_width=True):
                    toggle = f"{section_key}_{title}"
                    st.session_state.expanded_movie = None if st.session_state.expanded_movie==toggle else toggle
                    st.rerun()

                if genres:
                    st.markdown(render_genre_tags(genres), unsafe_allow_html=True)

                if m.get("confidence"):
                    rc = RISK_COLOR.get(m.get("risk","Medium"),"#ecc94b")
                    ee = EMOTION_EMOJI.get(m.get("emotion",""),"")
                    st.markdown(f"""<div style='font-size:11px;color:#888;margin-top:2px;'>
                        <span style='color:#63b3ed;font-weight:700;'>{m["confidence"]}%</span>
                        &nbsp;{ee} {m.get("emotion","")}
                        &nbsp;<span style='color:{rc};'>● {m.get("risk","")}</span>
                    </div>""", unsafe_allow_html=True)

        exp = next((t for t in row_t if st.session_state.expanded_movie==f"{section_key}_{t}"), None)
        if exp:
            show_detail_panel(exp, section_key, meta=(meta_dict or {}).get(exp))

# ─────────────────────────────────────────
# SEARCH + FILTER
# ─────────────────────────────────────────
st.markdown("<div class='section-header'>🔍 Search & Filter</div>", unsafe_allow_html=True)
sc, gc = st.columns([3,2])
with sc: sq = st.text_input("s", placeholder="Search by title…", label_visibility="collapsed")
with gc: sg = st.multiselect("g", ALL_GENRES, placeholder="🏷️ Filter by genre…", label_visibility="collapsed")

if sq or sg:
    sdf = movies_df.copy()
    if sq: sdf = sdf[sdf["title"].str.contains(sq, case=False, na=False)]
    if sg: sdf = sdf[sdf["genres"].apply(lambda g: all(t in str(g) for t in sg))]
    sdf = sdf.head(20)
    if sdf.empty:
        st.warning("No movies found.")
    else:
        st.markdown(f"<p style='color:#888;font-size:13px;margin-bottom:12px;'>Found {len(sdf)} movie(s)</p>",
                    unsafe_allow_html=True)
        show_grid(sdf, "search")
    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:24px 0;'>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────
# COUNT SLIDER
# ─────────────────────────────────────────
rec_count = st.slider("🎛️ Movies to show", 5, 30, 10, 1)

# ─────────────────────────────────────────
# ONE PERFECT RECOMMENDATION
# ─────────────────────────────────────────
st.markdown("<div class='section-header'>✨ One Perfect Pick</div>", unsafe_allow_html=True)
try:
    pdf, p_conf, p_risk = cached_perfect(uid)
    if not pdf.empty:
        pt = pdf.iloc[0]["title"]
        pg = cached_genres(pt)
        pp = cached_poster(pt)
        pc1, pc2 = st.columns([1,3])
        with pc1:
            if pp and os.path.exists(pp):
                try: st.image(Image.open(pp), use_container_width=True)
                except: pass
        with pc2:
            rc = RISK_COLOR.get(p_risk,"#ecc94b")
            st.markdown(f"""
            <div class='one-perfect-card'>
                <p style='color:#b794f4;font-size:11px;font-weight:700;margin-bottom:6px;letter-spacing:1.5px;'>✨ HANDPICKED FOR YOU</p>
                <h2 style='color:#e8e8ff;margin-bottom:8px;'>{pt}</h2>
                {render_genre_tags(pg)}
                <div style='margin-top:12px;'>
                  <div style='display:flex;justify-content:space-between;font-size:11px;color:#888;margin-bottom:4px;'>
                    <span>Confidence</span><span style='color:#63b3ed;font-weight:700;'>{p_conf}%</span>
                  </div>
                  <div class='conf-wrap'><div class='conf-fill' style='width:{p_conf}%;'></div></div>
                </div>
                <span class='meta-badge' style='background:{rc}22;color:{rc};border:1px solid {rc}44;margin-top:8px;'>Risk: {p_risk}</span>
            </div>""", unsafe_allow_html=True)
except Exception as e:
    st.info("Rate a few movies first to get your perfect pick!")

st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:20px 0;'>",
            unsafe_allow_html=True)

# ─────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────
st.markdown("<div class='section-header'>🎯 Recommended For You</div>", unsafe_allow_html=True)

rb1, rb2 = st.columns([1,1])
with rb1:
    if st.button("🔄 Refresh"):
        st.session_state.rec_seed += 1
        st.session_state.expanded_movie = None
        cached_recs.clear()
        st.rerun()
with rb2:
    if st.button("🎲 Surprise Me!"):
        st.session_state.rec_seed += 100
        st.session_state.expanded_movie = None
        st.session_state.surprise_mode = True
        cached_surprise.clear()
        st.rerun()

traits = cached_traits(uid)
if traits:
    st.markdown(
        '<div style="margin-bottom:12px;">' +
        "".join(f"<span class='trait-pill'>{TRAIT_EMOJI.get(t,'🎬')} {t}</span>" for t,_ in traits) +
        "</div>", unsafe_allow_html=True)

with st.spinner("Loading recommendations…"):
    try:
        if st.session_state.surprise_mode:
            st.session_state.surprise_mode = False
            srec = cached_surprise(uid, rec_count, st.session_state.rec_seed)
            st.info("🎲 Surprise mode — exploring outside your comfort zone!")
            show_grid(srec, "rec")
        else:
            rdf, rmeta, rtraits = cached_recs(uid, rec_count, st.session_state.rec_seed)
            show_grid(rdf, "rec", meta_dict=rmeta)
    except Exception as e:
        st.warning(f"Could not load recommendations: {e}")

# ─────────────────────────────────────────
# POPULAR MOVIES
# ─────────────────────────────────────────
st.markdown("<div class='section-header'>🔥 Popular Movies</div>", unsafe_allow_html=True)
if st.button("🔄 Refresh Popular"):
    st.session_state.pop_seed += 1
    st.session_state.expanded_movie = None
    cached_popular.clear()
    st.rerun()

with st.spinner("Loading popular movies…"):
    try:
        pop = cached_popular(rec_count, st.session_state.pop_seed)
        show_grid(pop, "pop")
    except Exception as e:
        st.warning(f"Could not load popular movies: {e}")
