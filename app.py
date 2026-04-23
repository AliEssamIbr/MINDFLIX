
import streamlit as st
import requests
from PIL import Image
import pandas as pd
import os, csv, re, random, base64
import numpy as np

from recommender import (
    recommend_with_metadata, get_popular_movies,
    get_personality_traits, get_taste_evolution,
    get_user_preferences, surprise_me,
    one_perfect_recommendation, global_avg_ratings,
    TRAIT_EMOJI, TRAIT_DESC, EMOTION_EMOJI,
)

API_KEY = "API KEY HERE"
st.set_page_config(page_title="AI Movie Recommender", layout="wide")

ALL_GENRES = ["Action","Adventure","Animation","Children","Comedy","Crime",
              "Documentary","Drama","Fantasy","Film-Noir","Horror","IMAX",
              "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]
RISK_COLOR    = {"Low":"#48bb78","Medium":"#ecc94b","High":"#fc8181"}
EMOTION_COLOR = {"Exciting":"rgba(252,129,74,0.18)",
                 "Thought-provoking":"rgba(99,179,237,0.18)",
                 "Light-hearted":"rgba(104,211,145,0.18)"}

def _logo_b64(path):
    if os.path.exists(path):
        with open(path,"rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

LOGO_B64 = _logo_b64("uni_logos_transparent.png") or _logo_b64("uni_logos.png")

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background-color:#080812;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cdefs%3E%3Cpattern id='g' width='120' height='120' patternUnits='userSpaceOnUse'%3E%3Cpolygon points='60,5 115,35 115,85 60,115 5,85 5,35' fill='none' stroke='%231a1a3a' stroke-width='1'/%3E%3Cline x1='60' y1='5' x2='60' y2='115' stroke='%2314143a' stroke-width='0.5'/%3E%3Cline x1='5' y1='35' x2='115' y2='85' stroke='%2314143a' stroke-width='0.5'/%3E%3Cline x1='115' y1='35' x2='5' y2='85' stroke='%2314143a' stroke-width='0.5'/%3E%3C/pattern%3E%3C/defs%3E%3Crect width='120' height='120' fill='url(%23g)'/%3E%3C/svg%3E");
    color:#f0f0f0;
}
.section-header{font-size:20px;font-weight:700;color:#e8e8f0;margin:28px 0 14px;padding-bottom:8px;border-bottom:2px solid rgba(99,179,237,0.35);}
.genre-tag{display:inline-block;background:rgba(255,255,255,0.08);color:#c8c8e0;border:1px solid rgba(255,255,255,0.14);border-radius:20px;padding:2px 10px;margin:2px;font-size:11px;font-weight:500;}
.trait-pill{display:inline-block;background:rgba(159,122,234,0.18);color:#c4a8ff;border:1px solid rgba(159,122,234,0.28);border-radius:20px;padding:5px 14px;margin:3px;font-size:13px;font-weight:600;}
.rated-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:13px;color:#bbb;}
.rated-score{background:rgba(99,179,237,0.18);color:#63b3ed;border-radius:20px;padding:2px 10px;font-weight:600;font-size:12px;white-space:nowrap;}
.conf-wrap{background:rgba(255,255,255,0.07);border-radius:20px;height:7px;width:100%;margin:5px 0;}
.conf-fill{height:7px;border-radius:20px;background:linear-gradient(90deg,#63b3ed,#9f7aea);}
.meta-badge{display:inline-block;border-radius:20px;padding:3px 12px;font-size:12px;font-weight:600;margin:2px 3px;}
.one-perfect-card{background:linear-gradient(135deg,rgba(20,20,60,0.95),rgba(40,20,70,0.95));border:2px solid rgba(159,122,234,0.45);border-radius:16px;padding:24px;box-shadow:0 0 40px rgba(159,122,234,0.15);}
.no-poster{height:200px;background:rgba(30,30,60,0.7);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:38px;border:1px solid rgba(255,255,255,0.06);}
.profile-card{background:rgba(255,255,255,0.04);border-radius:12px;padding:20px;border:1px solid rgba(255,255,255,0.08);margin-bottom:14px;}
/* modal backdrop */
.modal-backdrop{position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.82);z-index:8000;backdrop-filter:blur(6px);pointer-events:none;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# DATA + CACHE
# ─────────────────────────────────────────
movies_df = pd.read_csv("movies_processed.csv")

@st.cache_data(show_spinner=False)
def cached_poster(title):
    fp = f"posters/{re.sub(r'[\\/*?:\"<>|]','',title).replace(' ','_')}.jpg"
    if os.path.exists(fp): return fp
    try:
        d = requests.get("https://api.themoviedb.org/3/search/movie",
            params={"api_key":API_KEY,"query":re.sub(r"\(\d{4}\)","",title).strip()},
            timeout=5).json()
        if d.get("results"):
            pp = d["results"][0].get("poster_path")
            if pp:
                img = requests.get(f"https://image.tmdb.org/t/p/w500{pp}",timeout=5).content
                with open(fp,"wb") as f: f.write(img)
                return fp
    except: pass
    return None

@st.cache_data(show_spinner=False)
def cached_poster_b64(title):
    fp = cached_poster(title)
    if fp and os.path.exists(fp):
        with open(fp,"rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

@st.cache_data(show_spinner=False)
def cached_genres(title):
    row = movies_df[movies_df["title"]==title]
    if row.empty: return []
    g = row.iloc[0].get("genres","")
    if pd.isna(g) or not g: return []
    return [x.strip() for x in str(g).split("|") if x.strip() and x.strip()!="(no genres listed)"]

@st.cache_data(ttl=30,show_spinner=False)
def cached_recs(user_id,top_k,seed):
    random.seed(seed)
    b = recommend_with_metadata(user_id, top_k=top_k+15)
    titles = list(b["df"]["title"]); random.shuffle(titles); chosen=titles[:top_k]
    return (b["df"][b["df"]["title"].isin(chosen)],
            {t:b["meta"][t] for t in chosen if t in b["meta"]},
            b.get("traits",[]))

@st.cache_data(ttl=30,show_spinner=False)
def cached_popular(top_k,seed):
    np.random.seed(seed); return get_popular_movies(top_k)

@st.cache_data(ttl=30,show_spinner=False)
def cached_surprise(user_id,top_k,seed):
    np.random.seed(seed); return surprise_me(user_id,top_k)

@st.cache_data(ttl=60,show_spinner=False)
def cached_perfect(user_id): return one_perfect_recommendation(user_id)

@st.cache_data(ttl=60,show_spinner=False)
def cached_traits(user_id): return get_personality_traits(user_id)

@st.cache_data(ttl=60,show_spinner=False)
def cached_evolution(user_id): return get_taste_evolution(user_id)

@st.cache_data(ttl=30,show_spinner=False)
def cached_user_ratings(user_id):
    if not os.path.exists("user_data.csv"): return []
    df = pd.read_csv("user_data.csv"); df["userId"]=df["userId"].astype(str)
    ur = df[df["userId"]==str(user_id)].merge(
        movies_df[["movieId","title"]],on="movieId",how="left").dropna(subset=["title"])
    return list(zip(ur["title"],ur["rating"]))[::-1]

def invalidate_user_cache():
    cached_recs.clear(); cached_perfect.clear()
    cached_traits.clear(); cached_evolution.clear(); cached_user_ratings.clear()

def clean_title(t): return re.sub(r"\(\d{4}\)","",t).strip()
def star_label(r):  return "⭐"*int(r)+("½" if r%1>=0.5 else "")

def render_genre_tags(genres):
    if not genres: return ""
    return ('<div style="padding:4px 0 6px 0;">'
            +"".join(f'<span class="genre-tag">{g}</span>' for g in genres)+"</div>")

def save_rating(user_id,title,rating):
    row=movies_df[movies_df["title"]==title]
    if row.empty: return False
    mid=row.iloc[0]["movieId"]
    exists=os.path.isfile("user_data.csv")
    with open("user_data.csv","a",newline="") as f:
        w=csv.writer(f)
        if not exists: w.writerow(["userId","movieId","rating"])
        w.writerow([user_id,mid,rating])
    return True

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
for k,v in {
    "user_id":None,"logged_in":False,
    "modal_movie":None,"modal_section":None,"modal_meta":None,
    "pending_rating":{},"rating_submitted":set(),
    "rec_seed":0,"pop_seed":0,"show_profile":False,
    "surprise_mode":False,"perfect_submitted":False,
    "_last_perfect_uid":None,
}.items():
    if k not in st.session_state: st.session_state[k]=v

# Reset perfect_submitted and stale ratings whenever the user changes.
# This was the root cause of the "stuck" Perfect Pick button.
if st.session_state["_last_perfect_uid"] != st.session_state.get("user_id"):
    st.session_state["perfect_submitted"] = False
    st.session_state["_last_perfect_uid"] = st.session_state.get("user_id")
    st.session_state["rating_submitted"]  = set()
    st.session_state["pending_rating"]    = {}

# ═══════════════════════════════════════════
# LOGIN PAGE
# ═══════════════════════════════════════════
if not st.session_state.logged_in:

    if LOGO_B64:
        st.markdown(
            f'<div style="display:flex;justify-content:center;margin-top:36px;margin-bottom:4px;">'
            f'<img src="data:image/png;base64,{LOGO_B64}" style="width:420px;max-width:88vw;"/>'
            f'</div>',
            unsafe_allow_html=True)

    st.markdown(
        '<div style="text-align:center;padding:10px 20px 8px;">'
        '<h1 style="color:#e8e8ff;font-size:34px;margin-bottom:6px;">AI Movie Recommender</h1>'
        '<p style="color:#9090bb;font-size:15px;margin-bottom:4px;font-weight:500;">University of Baghdad</p>'
        '<p style="color:#7070aa;font-size:13px;margin-bottom:0;">College of Artificial Intelligence &nbsp;·&nbsp; Big Data Department</p>'
        '</div>',
        unsafe_allow_html=True)

    # Team card — built with pure st calls to avoid raw HTML rendering bug
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    _, card_col, _ = st.columns([1, 1.4, 1])
    with card_col:
        with st.container():
            st.markdown(
                '<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.09);'
                'border-radius:16px;padding:22px 44px;text-align:center;">'
                '<p style="color:#8888aa;font-size:11px;letter-spacing:2px;margin-bottom:8px;text-transform:uppercase;">Supervised By</p>'
                '<p style="color:#c4a8ff;font-size:17px;font-weight:700;margin-bottom:18px;">Anfal Mudhaffar Ali</p>'
                '<p style="color:#8888aa;font-size:11px;letter-spacing:2px;margin-bottom:10px;text-transform:uppercase;">Students</p>'
                '<p style="color:#dcdcff;font-size:15px;line-height:2.1;margin:0;font-weight:500;">'
                'Amir Muntaser Hussein<br>Mohammed Salah Mahdi<br>Ali Essam Ibrahim'
                '</p></div>',
                unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    _, login_col, _ = st.columns([1, 1.1, 1])
    with login_col:
        st.markdown(
            '<p style="color:#aaaacc;text-align:center;font-size:15px;font-weight:600;margin-bottom:8px;">Choose your profile</p>',
            unsafe_allow_html=True)
        uid_in = st.number_input("uid", min_value=1, step=1, label_visibility="collapsed")
        if st.button("▶  Log In", use_container_width=True):
            st.session_state.user_id = int(uid_in)
            st.session_state.logged_in = True
            st.rerun()
        st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:14px 0;'>", unsafe_allow_html=True)
        st.markdown('<p style="color:#555;text-align:center;font-size:12px;">New here?</p>', unsafe_allow_html=True)
        if st.button("🎲  Create New User", use_container_width=True):
            nid = 1
            if os.path.exists("user_data.csv"):
                df = pd.read_csv("user_data.csv")
                if not df.empty: nid = int(df["userId"].max()) + 1
            st.session_state.user_id = nid
            st.session_state.logged_in = True
            st.rerun()
    st.stop()

# ═══════════════════════════════════════════
# MODAL — UNIFIED SINGLE CARD
# Layout (matches sketch):
#   [vertical poster — tall rectangle]
#   [title / year / avg]
#   [genre tags]
#   [emotion badge + reason]
#   [confidence bar]      ← labelled "Confidence"
#   ─────────────────────
#   [rating slider]       ← labelled "Rate"
#   [Submit]   [Close]
# ═══════════════════════════════════════════
uid = st.session_state.user_id

if st.session_state.modal_movie:
    title   = st.session_state.modal_movie
    sec     = st.session_state.modal_section or "modal"
    meta    = st.session_state.modal_meta or {}
    genres  = cached_genres(title)
    year_m  = re.search(r"\((\d{4})\)", title)
    year    = year_m.group(1) if year_m else "N/A"
    clean   = clean_title(title)
    rkey    = f"rating_{sec}_{title}"

    mid_row  = movies_df[movies_df["title"]==title]
    glob_avg = 3.0
    if not mid_row.empty:
        glob_avg = round(global_avg_ratings.get(mid_row.iloc[0]["movieId"],3.0),1)

    conf    = meta.get("confidence", 0)
    risk    = meta.get("risk", "")
    emotion = meta.get("emotion", "")
    reason  = meta.get("reason", "")
    ee      = EMOTION_EMOJI.get(emotion, "")
    ec      = EMOTION_COLOR.get(emotion, "rgba(255,255,255,0.06)")
    rc      = RISK_COLOR.get(risk, "#aaa") if risk else ""

    poster_b64 = cached_poster_b64(title)

    genre_html = render_genre_tags(genres)

    # ── centred card column ──
    _, mcol, _ = st.columns([1, 2, 1])
    with mcol:

        # ── Outer card wrapper opens ──
        st.markdown(
            f"<div style='"
            f"background:linear-gradient(170deg,#0c0c22,#13132e,#0f0f28);"
            f"border:1px solid rgba(99,179,237,0.30);border-radius:20px;"
            f"padding:22px 24px 4px;"
            f"box-shadow:0 20px 60px rgba(0,0,0,0.7);"
            f"'>",
            unsafe_allow_html=True
        )

        # ── TWO-COLUMN layout: poster LEFT, info RIGHT ──
        img_col, info_col = st.columns([1, 1.6])

        with img_col:
            if poster_b64:
                st.markdown(
                    f"<div style='width:100%;overflow:hidden;border-radius:12px;"
                    f"position:relative;height:100%;min-height:280px;'>"
                    f"<img src='data:image/jpeg;base64,{poster_b64}' "
                    f"style='width:100%;height:100%;object-fit:cover;object-position:center top;"
                    f"display:block;filter:brightness(0.88);border-radius:12px;'/>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='width:100%;min-height:280px;background:rgba(20,20,50,0.9);"
                    f"display:flex;align-items:center;justify-content:center;"
                    f"font-size:64px;border-radius:12px;'>🎞️</div>",
                    unsafe_allow_html=True
                )

        with info_col:
            # Title
            st.markdown(
                f"<div style='color:#eeeeff;font-size:20px;font-weight:700;"
                f"line-height:1.3;margin-bottom:6px;'>{title}</div>",
                unsafe_allow_html=True
            )
            # Year · Global avg
            st.markdown(
                f"<div style='color:#666;font-size:12px;margin-bottom:10px;'>"
                f"📅 {year} &nbsp;·&nbsp; "
                f"🌍 Global avg: <strong style='color:#aaa;'>{glob_avg} ⭐</strong></div>",
                unsafe_allow_html=True
            )
            # Genre tags
            st.markdown(genre_html, unsafe_allow_html=True)

            # Emotion badge + confidence % on same line
            if emotion:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:center;margin:8px 0 4px;'>"
                    f"<span class='meta-badge' style='background:{ec};color:#e8e8f0;"
                    f"border:1px solid rgba(255,255,255,0.1);margin:0;'>{ee} {emotion}</span>"
                    f"<span style='color:#63b3ed;font-weight:700;font-size:13px;'>{conf}%</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # Confidence label + bar
            if conf:
                st.markdown(
                    f"<div style='margin:2px 0 8px;'>"
                    f"<div style='font-size:11px;color:#777;margin-bottom:4px;'>"
                    f"Confidence</div>"
                    f"<div class='conf-wrap'>"
                    f"<div class='conf-fill' style='width:{conf}%;'></div>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

            # Reason line
            if reason:
                st.markdown(
                    f"<p style='color:#8888aa;font-size:12px;margin:4px 0 8px;"
                    f"line-height:1.5;'>💡 {reason}</p>",
                    unsafe_allow_html=True
                )

        # ── Divider ──
        st.markdown(
            "<hr style='border:none;border-top:1px solid rgba(99,179,237,0.15);margin:16px 0 10px;'/>",
            unsafe_allow_html=True
        )

        # ── Rating slider label: "⭐ Your rating for <clean title>" ──
        st.markdown(
            f"<p style='color:#e8c84a;font-size:13px;font-weight:600;margin:0 0 4px;'>"
            f"⭐ Your rating for {clean}</p>",
            unsafe_allow_html=True
        )

        # ── SLIDER — must be a Streamlit widget ──
        prev      = next((r for t, r in cached_user_ratings(uid) if t == title), None)
        default_r = float(prev) if prev else st.session_state.pending_rating.get(rkey, 3.0)

        new_r = st.slider(
            "rate_slider",
            min_value=0.5, max_value=5.0,
            value=default_r, step=0.5,
            key=f"modal_slider_{title}",
            label_visibility="collapsed"
        )
        st.session_state.pending_rating[rkey] = new_r

        # ── BUTTONS row ──
        b_left, b_right = st.columns(2)
        with b_left:
            if rkey in st.session_state.rating_submitted:
                st.success(f"✅ Saved {new_r}⭐")
            else:
                if st.button("✅  Submit Rating", use_container_width=True,
                             key=f"modal_submit_{title}"):
                    if save_rating(uid, title, new_r):
                        st.session_state.rating_submitted.add(rkey)
                        invalidate_user_cache()
                        st.rerun()
        with b_right:
            if st.button("✖  Close", use_container_width=True, key="modal_close"):
                st.session_state.modal_movie   = None
                st.session_state.modal_section = None
                st.session_state.modal_meta    = None
                st.rerun()

        # close the outer card div
        st.markdown(
            "<div style='height:20px;'></div></div>"
            "<div style='height:40px;'></div>",
            unsafe_allow_html=True
        )

    st.stop()

# ═══════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════
L, R = st.columns([5, 3])
with L:
    st.markdown("<h2 style='color:#e8e8ff;margin:0;'>🎬 AI Movie Recommender</h2>",
                unsafe_allow_html=True)
with R:
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("👤 Profile"):
            st.session_state.show_profile = not st.session_state.show_profile
    with c2:
        st.markdown(f"<p style='color:#666;padding-top:8px;text-align:center;"
                    f"font-size:13px;'>User {uid}</p>", unsafe_allow_html=True)
    with c3:
        if st.button("🚪 Logout"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin-bottom:4px;'>",
            unsafe_allow_html=True)

# ─────────────────────────────────────────
# PROFILE PANEL
# ─────────────────────────────────────────
if st.session_state.show_profile:
    rated            = cached_user_ratings(uid)
    traits           = cached_traits(uid)
    early_g, recent_g = cached_evolution(uid)

    st.markdown(
        f"<div class='profile-card'>"
        f"<h3 style='color:#e0e0ff;margin-bottom:4px;'>👤 User {uid}</h3>"
        f"<p style='color:#666;font-size:13px;'>Ratings: "
        f"<strong style='color:#63b3ed;'>{len(rated)}</strong></p></div>",
        unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎬 My Ratings", "🧠 Personality", "📈 Taste Evolution"])

    with tab1:
        if not rated:
            st.info("No ratings yet — click any movie poster to rate it!")
        else:
            seen = {}
            for t, r in rated:
                if t not in seen: seen[t] = r
            rows_html = ""
            for t, r in list(seen.items())[:40]:
                mr = movies_df[movies_df["title"]==t]
                ga = 3.0
                if not mr.empty: ga = round(global_avg_ratings.get(mr.iloc[0]["movieId"],3.0),1)
                diff = round(float(r)-ga,1)
                ds = ""
                if diff > 0.5:   ds = f"<span style='color:#48bb78;font-size:10px;'> +{diff} vs crowd</span>"
                elif diff < -0.5: ds = f"<span style='color:#fc8181;font-size:10px;'> {diff} vs crowd</span>"
                short = t[:44]+"…" if len(t)>44 else t
                rows_html += (f"<div class='rated-row'><span>{short}{ds}</span>"
                              f"<span class='rated-score'>{star_label(r)} {r}</span></div>")
            st.markdown(rows_html, unsafe_allow_html=True)
            st.markdown("#### 🎭 Genre Preferences")
            gs = {}
            for t, r in seen.items():
                for g in cached_genres(t): gs.setdefault(g,[]).append(float(r))
            for g, sc in sorted({g:round(sum(v)/len(v),2) for g,v in gs.items()}.items(),
                                key=lambda x: -x[1])[:8]:
                st.progress(sc/5.0, text=f"{g}  —  {sc} ⭐")

    with tab2:
        if not traits:
            st.info("Rate at least 3 movies to unlock your personality profile!")
        else:
            st.markdown("<h4 style='color:#e0e0ff;margin-bottom:12px;'>Your Cinematic Personality</h4>",
                        unsafe_allow_html=True)
            pills = "".join(
                f"<span class='trait-pill'>{TRAIT_EMOJI.get(t,'🎬')} {t}</span>"
                for t,_,_ in traits)
            st.markdown(f"<div style='margin-bottom:16px;'>{pills}</div>",
                        unsafe_allow_html=True)
            for t, count, avg_r in traits:
                st.markdown(
                    f"<div style='background:rgba(159,122,234,0.08);border-radius:10px;"
                    f"padding:12px 16px;margin-bottom:8px;"
                    f"border:1px solid rgba(159,122,234,0.18);'>"
                    f"<strong style='color:#b794f4;'>{TRAIT_EMOJI.get(t,'')} {t}</strong>"
                    f"<span style='color:#555;font-size:12px;margin-left:8px;'>"
                    f"{int(count)} movies · avg {avg_r}⭐</span>"
                    f"<p style='color:#aaa;font-size:13px;margin:4px 0 0;'>{TRAIT_DESC.get(t,'')}</p>"
                    f"</div>",
                    unsafe_allow_html=True)

    with tab3:
        if not early_g or not recent_g:
            st.info("Rate more movies over time to see your taste evolution!")
        else:
            ce, ca, cr = st.columns([2,1,2])
            with ce:
                st.markdown(
                    f"<div style='background:rgba(99,179,237,0.08);border-radius:12px;"
                    f"padding:20px;border:1px solid rgba(99,179,237,0.18);text-align:center;'>"
                    f"<p style='color:#666;font-size:11px;margin-bottom:4px;letter-spacing:1px;'>EARLY TASTE</p>"
                    f"<h3 style='color:#63b3ed;margin:0;'>{early_g}</h3></div>",
                    unsafe_allow_html=True)
            with ca:
                st.markdown("<div style='font-size:28px;text-align:center;color:#63b3ed;padding:14px;'>→</div>",
                            unsafe_allow_html=True)
            with cr:
                st.markdown(
                    f"<div style='background:rgba(159,122,234,0.08);border-radius:12px;"
                    f"padding:20px;border:1px solid rgba(159,122,234,0.18);text-align:center;'>"
                    f"<p style='color:#666;font-size:11px;margin-bottom:4px;letter-spacing:1px;'>NOW</p>"
                    f"<h3 style='color:#b794f4;margin:0;'>{recent_g}</h3></div>",
                    unsafe_allow_html=True)
            msg = ("Consistent taste — you know what you like! 🎯"
                   if early_g==recent_g
                   else f"Evolved from {early_g} → {recent_g}")
            st.markdown(f"<p style='color:#666;text-align:center;margin-top:12px;font-size:13px;'>{msg}</p>",
                        unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:20px 0;'>",
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
                    st.session_state.modal_movie   = title
                    st.session_state.modal_section = section_key
                    st.session_state.modal_meta    = m
                    st.rerun()

                if genres:
                    st.markdown(render_genre_tags(genres), unsafe_allow_html=True)
                if m.get("confidence"):
                    ee = EMOTION_EMOJI.get(m.get("emotion",""),"")
                    st.markdown(
                        f"<div style='font-size:11px;color:#666;margin-top:2px;'>"
                        f"<span style='color:#63b3ed;font-weight:700;'>{m['confidence']}%</span>"
                        f"&nbsp;{ee} {m.get('emotion','')}</div>",
                        unsafe_allow_html=True)

# ─────────────────────────────────────────
# SEARCH & FILTER
# ─────────────────────────────────────────
st.markdown("<div class='section-header'>🔍 Search & Filter</div>", unsafe_allow_html=True)
sc, gc = st.columns([3,2])
with sc: sq = st.text_input("s", placeholder="Search by title…", label_visibility="collapsed")
with gc: sg = st.multiselect("g", ALL_GENRES, placeholder="🏷️ Filter by genre…",
                              label_visibility="collapsed")
if sq or sg:
    sdf = movies_df.copy()
    if sq: sdf = sdf[sdf["title"].str.contains(sq, case=False, na=False)]
    if sg: sdf = sdf[sdf["genres"].apply(lambda g: all(t in str(g) for t in sg))]
    sdf = sdf.head(20)
    if sdf.empty: st.warning("No movies found.")
    else:
        st.markdown(f"<p style='color:#666;font-size:13px;margin-bottom:12px;'>"
                    f"Found {len(sdf)} movie(s)</p>", unsafe_allow_html=True)
        show_grid(sdf, "search")
    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:24px 0;'>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────
# COUNT SLIDER
# ─────────────────────────────────────────
rec_count = st.slider("🎛️ Movies to show", 5, 30, 10, 1)

# ─────────────────────────────────────────
# ONE PERFECT PICK
# ─────────────────────────────────────────
st.markdown("<div class='section-header'>✨ One Perfect Pick</div>", unsafe_allow_html=True)
try:
    pdf, p_conf, p_risk = cached_perfect(uid)
    if not pdf.empty:
        pt      = pdf.iloc[0]["title"]
        pg      = cached_genres(pt)
        pp_path = cached_poster(pt)
        rc      = RISK_COLOR.get(p_risk, "#ecc94b")
        pc1, pc2 = st.columns([1,3])
        with pc1:
            if pp_path and os.path.exists(pp_path):
                try: st.image(Image.open(pp_path), use_container_width=True)
                except: pass
        with pc2:
            st.markdown(
                f"<div class='one-perfect-card'>"
                f"<p style='color:#b794f4;font-size:11px;font-weight:700;"
                f"margin-bottom:6px;letter-spacing:1.5px;'>✨ HANDPICKED FOR YOU</p>"
                f"<h2 style='color:#e8e8ff;margin-bottom:8px;'>{pt}</h2>"
                f"{render_genre_tags(pg)}"
                f"<div style='margin-top:12px;'>"
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:11px;color:#888;margin-bottom:4px;'>"
                f"<span>Confidence</span>"
                f"<span style='color:#63b3ed;font-weight:700;'>{p_conf}%</span></div>"
                f"<div class='conf-wrap'><div class='conf-fill' style='width:{p_conf}%;'></div></div>"
                f"</div></div>",
                unsafe_allow_html=True)

            p_rkey  = f"rating_perfect_{pt}"
            p_prev  = next((r for t,r in cached_user_ratings(uid) if t==pt), None)
            p_def   = float(p_prev) if p_prev else st.session_state.pending_rating.get(p_rkey, 3.0)
            p_new_r = st.slider("⭐ Rate this pick", 0.5, 5.0, p_def, 0.5, key="perfect_slider")
            st.session_state.pending_rating[p_rkey] = p_new_r
            if p_rkey in st.session_state.rating_submitted or st.session_state.perfect_submitted:
                st.success(f"✅ Rated {p_new_r}⭐")
            else:
                if st.button("Submit Rating", key="perfect_submit"):
                    if save_rating(uid, pt, p_new_r):
                        st.session_state.rating_submitted.add(p_rkey)
                        st.session_state.perfect_submitted = True
                        invalidate_user_cache(); st.rerun()
except:
    st.info("Rate a few movies first to get your perfect pick!")

st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:20px 0;'>",
            unsafe_allow_html=True)

# ─────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────
st.markdown("<div class='section-header'>🎯 Recommended For You</div>", unsafe_allow_html=True)

traits = cached_traits(uid)
if traits:
    pills = "".join(
        f"<span class='trait-pill'>{TRAIT_EMOJI.get(t,'🎬')} {t}</span>"
        for t,_,_ in traits)
    st.markdown(f"<div style='margin-bottom:12px;'>{pills}</div>", unsafe_allow_html=True)

rb1, rb2 = st.columns(2)
with rb1:
    if st.button("🔄 Refresh Recommendations"):
        st.session_state.rec_seed += 1; cached_recs.clear(); st.rerun()
with rb2:
    if st.button("🎲 Surprise Me!"):
        st.session_state.rec_seed += 100
        st.session_state.surprise_mode = True
        cached_surprise.clear(); st.rerun()

with st.spinner("Loading…"):
    try:
        if st.session_state.surprise_mode:
            st.session_state.surprise_mode = False
            sr = cached_surprise(uid, rec_count, st.session_state.rec_seed)
            st.info("🎲 Exploring outside your comfort zone!")
            show_grid(sr, "rec")
        else:
            rdf, rmeta, _ = cached_recs(uid, rec_count, st.session_state.rec_seed)
            show_grid(rdf, "rec", meta_dict=rmeta)
    except Exception as e:
        st.warning(f"Could not load recommendations: {e}")

# ─────────────────────────────────────────
# POPULAR
# ─────────────────────────────────────────
st.markdown("<div class='section-header'>🔥 Popular Movies</div>", unsafe_allow_html=True)
if st.button("🔄 Refresh Popular"):
    st.session_state.pop_seed += 1; cached_popular.clear(); st.rerun()

with st.spinner("Loading…"):
    try:
        pop = cached_popular(rec_count, st.session_state.pop_seed)
        show_grid(pop, "pop")
    except Exception as e:
        st.warning(f"Could not load popular movies: {e}")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown(
    "<hr style='border-color:rgba(255,255,255,0.06);margin-top:48px;'>"
    "<div style='text-align:center;padding:20px 0 32px;'>"
    "<p style='color:#2e2e50;font-size:12px;margin-bottom:4px;'>"
    "University of Baghdad · College of Artificial Intelligence · Big Data Department</p>"
    "<p style='color:#252540;font-size:11px;'>"
    "Supervised by Anfal Mudhaffar Ali · "
    "Ali Essam Ibrahim · Mohammed Salah Mahdi · Amir Muntaser Hussein</p>"
    "</div>",
    unsafe_allow_html=True)
