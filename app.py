# app.py — Influencer Detection
import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Influencer Detection",
    page_icon="📸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Minimal CSS override ───────────────────────────────────────
st.markdown("""
<style>
/* Tighten up default Streamlit padding */
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 680px; }

/* Section labels */
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(128,128,128,0.8);
    margin-bottom: 0.25rem;
}

/* Result card */
.result-card {
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-top: 1rem;
}
.result-type {
    font-size: 20px;
    font-weight: 500;
    margin: 0 0 4px;
}
.result-sub {
    font-size: 13px;
    color: rgba(128,128,128,0.8);
    margin: 0 0 1rem;
}

/* Badges */
.badge {
    display: inline-block;
    font-size: 12px;
    font-weight: 500;
    padding: 3px 12px;
    border-radius: 6px;
    margin-bottom: 1rem;
}
.badge-fake    { background: #FCEBEB; color: #A32D2D; }
.badge-real    { background: #E1F5EE; color: #0F6E56; }
.badge-growing { background: #E6F1FB; color: #185FA5; }
.badge-normal  { background: #F1EFE8; color: #5F5E5A; }

/* Stat row */
.stat-row { display: flex; gap: 12px; margin-top: 0.75rem; }
.stat-box {
    flex: 1;
    background: rgba(128,128,128,0.06);
    border-radius: 8px;
    padding: 10px 14px;
}
.stat-val { font-size: 17px; font-weight: 500; margin: 0 0 2px; }
.stat-lbl { font-size: 11px; color: rgba(128,128,128,0.7); margin: 0; }

/* Confidence bar */
.bar-wrap { margin: 12px 0 4px; }
.bar-label {
    display: flex; justify-content: space-between;
    font-size: 12px; color: rgba(128,128,128,0.7);
    margin-bottom: 5px;
}
.bar-bg {
    height: 4px;
    background: rgba(128,128,128,0.12);
    border-radius: 2px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ── Load model artefacts ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    rf            = joblib.load('model_fixed/rf_model.pkl')
    scaler        = joblib.load('model_fixed/scaler.pkl')
    features      = joblib.load('model_fixed/feature_names.pkl')
    class_names   = joblib.load('model_fixed/class_names.pkl')
    fake_thresh   = joblib.load('model_fixed/fake_threshold.pkl')
    fake_idx      = joblib.load('model_fixed/fake_class_idx.pkl')
    return rf, scaler, features, class_names, fake_thresh, fake_idx

rf, scaler, FEATURES, class_names, fake_thresh, fake_idx = load_artifacts()

BADGE = {
    'Real Influencer':    ('Real',    'badge-real'),
    'Growing Influencer': ('Growing', 'badge-growing'),
    'Normal User':        ('Normal',  'badge-normal'),
    'Fake Influencer':    ('Fake / bot', 'badge-fake'),
}
STATUS_TEXT = {
    'Real Influencer':    'Real influencer',
    'Growing Influencer': 'Growing influencer',
    'Normal User':        'Not an influencer',
    'Fake Influencer':    'Fake / bot account',
}
BAR_COLOR = {
    'Real Influencer':    '#1D9E75',
    'Growing Influencer': '#378ADD',
    'Normal User':        '#888780',
    'Fake Influencer':    '#E24B4A',
}


# ── Header ─────────────────────────────────────────────────────
st.markdown("## Influencer detection")
st.caption("B.Tech 4th semester · ML project")
st.markdown("---")


# ── Inputs ─────────────────────────────────────────────────────
st.markdown('<p class="section-label">Account metrics</p>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    followers          = st.number_input("Followers",           50, 2_000_000, value=10_000, step=1000)
    following          = st.number_input("Following",           30,    50_000, value=500,    step=100)
    posts              = st.number_input("Total posts",          1,    50_000, value=100,    step=10)
with c2:
    avg_posts_per_day  = st.number_input("Avg posts per day",  0.0,     100.0, value=1.0,   step=0.1, format="%.1f")
    avg_views          = st.number_input("Avg views per post",   0, 5_000_000, value=5000,  step=100)
    account_age        = st.number_input("Account age (months)",1.0,     100.0, value=24.0, step=0.5, format="%.1f")

st.markdown('<p class="section-label" style="margin-top:0.75rem;">Engagement</p>', unsafe_allow_html=True)

c3, c4, c5 = st.columns(3)
with c3:
    avg_likes    = st.number_input("Avg likes per post",    0, 1_000_000, value=500,  step=100)
with c4:
    avg_comments = st.number_input("Avg comments per post", 0,   500_000, value=50,   step=10)
with c5:
    avg_shares   = st.number_input("Avg shares per post",   0,   500_000, value=30,   step=10)

st.markdown("---")


# ── Predict ────────────────────────────────────────────────────
if st.button("Detect influencer type", type="primary", use_container_width=True):

    EPS = 1

    # Engineered features (must match training pipeline exactly)
    likes_per_follower  = avg_likes    / (followers + EPS)
    views_per_follower  = avg_views    / (followers + EPS)
    likes_per_view      = avg_likes    / (avg_views + EPS)
    comments_per_like   = avg_comments / (avg_likes + EPS)
    shares_per_like     = avg_shares   / (avg_likes + EPS)
    follower_ratio      = followers    / (following + EPS)
    posting_efficiency  = posts        / (account_age + EPS)
    log_followers       = np.log1p(followers)
    log_views           = np.log1p(avg_views)
    engagement_rate     = ((avg_likes + avg_comments + avg_shares) / (followers + EPS)) * 100

    row = [
        followers, following, posts, avg_posts_per_day,
        avg_views, avg_likes, avg_comments, avg_shares, account_age,
        likes_per_follower, views_per_follower, likes_per_view,
        comments_per_like, shares_per_like,
        follower_ratio, posting_efficiency,
        log_followers, log_views,
    ]

    X_scaled   = scaler.transform(np.array([row]))
    proba      = rf.predict_proba(X_scaled)[0]
    pred_idx   = int(rf.predict(X_scaled)[0])

    # Apply tuned threshold for Fake class
    if proba[fake_idx] >= fake_thresh:
        pred_idx = fake_idx

    predicted_label = class_names[pred_idx]
    confidence      = float(proba[pred_idx]) * 100
    badge_text, badge_cls = BADGE[predicted_label]
    bar_color = BAR_COLOR[predicted_label]

    # ── Result card ────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card">
      <span class="badge {badge_cls}">{badge_text}</span>
      <p class="result-type">{STATUS_TEXT[predicted_label]}</p>
      <p class="result-sub">Predicted by Random Forest with tuned threshold</p>

      <div class="bar-wrap">
        <div class="bar-label">
          <span>Confidence</span>
          <span>{confidence:.1f}%</span>
        </div>
        <div class="bar-bg">
          <div style="height:4px; width:{confidence:.1f}%; background:{bar_color}; border-radius:2px;"></div>
        </div>
      </div>

      <div class="stat-row">
        <div class="stat-box">
          <p class="stat-val">{engagement_rate:.2f}%</p>
          <p class="stat-lbl">Engagement rate</p>
        </div>
        <div class="stat-box">
          <p class="stat-val">{likes_per_follower*100:.2f}%</p>
          <p class="stat-lbl">Likes / follower</p>
        </div>
        <div class="stat-box">
          <p class="stat-val">{views_per_follower*100:.2f}%</p>
          <p class="stat-lbl">Views / follower</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)