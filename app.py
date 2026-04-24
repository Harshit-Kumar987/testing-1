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

@st.cache_resource
def load_artifacts():
    rf          = joblib.load('model_fixed/rf_model.pkl')
    scaler      = joblib.load('model_fixed/scaler.pkl')
    features    = joblib.load('model_fixed/feature_names.pkl')
    class_names = joblib.load('model_fixed/class_names.pkl')
    fake_thresh = joblib.load('model_fixed/fake_threshold.pkl')
    fake_idx    = joblib.load('model_fixed/fake_class_idx.pkl')
    return rf, scaler, features, class_names, fake_thresh, fake_idx

rf, scaler, FEATURES, class_names, fake_thresh, fake_idx = load_artifacts()

STATUS = {
    'Real Influencer'    : 'Influencer',
    'Growing Influencer' : 'Growing Influencer',
    'Normal User'        : 'Not Influencer',
    'Fake Influencer'    : 'Fake / Bot Account',
}

# ── Header ─────────────────────────────────────────────────────
st.title("📸 Influencer Detection")
st.caption("B.Tech 4th Semester · ML Project")
st.divider()

# ── Inputs ─────────────────────────────────────────────────────
st.subheader("Enter Account Details")

c1, c2 = st.columns(2)

with c1:
    followers             = st.number_input("Followers",              50, 2_000_000, value=10_000,  step=1000)
    following             = st.number_input("Following",              30,    50_000, value=500,     step=100)
    posts                 = st.number_input("Total Posts",             1,    50_000, value=100,     step=10)
    avg_posts_per_day     = st.number_input("Avg Posts Per Day",     0.0,     100.0, value=1.0,    step=0.1)
    avg_views_per_post    = st.number_input("Avg Views Per Post",      0, 5_000_000, value=5000,   step=100)
    account_age_months    = st.number_input("Account Age (Months)",  1.0,     100.0, value=24.0,   step=0.5)

with c2:
    avg_likes_per_post    = st.number_input("Avg Likes Per Post",      0, 1_000_000, value=500,    step=100)
    avg_comments_per_post = st.number_input("Avg Comments Per Post",   0,   500_000, value=50,     step=10)
    avg_shares_per_post   = st.number_input("Avg Shares Per Post",     0,   500_000, value=30,     step=10)

st.divider()

# ── Predict ────────────────────────────────────────────────────
if st.button("Detect Influencer Type", type="primary", use_container_width=True):

    EPS = 1

    # Engineered features — must match training pipeline exactly
    likes_per_follower  = avg_likes_per_post    / (followers + EPS)
    views_per_follower  = avg_views_per_post    / (followers + EPS)
    likes_per_view      = avg_likes_per_post    / (avg_views_per_post + EPS)
    comments_per_like   = avg_comments_per_post / (avg_likes_per_post + EPS)
    shares_per_like     = avg_shares_per_post   / (avg_likes_per_post + EPS)
    follower_ratio      = followers             / (following + EPS)
    posting_efficiency  = posts                 / (account_age_months + EPS)
    log_followers       = np.log1p(followers)
    log_views           = np.log1p(avg_views_per_post)
    engagement_rate     = ((avg_likes_per_post + avg_comments_per_post + avg_shares_per_post) / (followers + EPS)) * 100
    likes_views_ratio   = avg_likes_per_post / (avg_views_per_post + 1)

    input_scaled = scaler.transform(np.array([[
        followers, following, posts,
        avg_posts_per_day, avg_views_per_post,
        avg_likes_per_post, avg_comments_per_post,
        avg_shares_per_post, account_age_months,
        likes_per_follower, views_per_follower, likes_per_view,
        comments_per_like, shares_per_like,
        follower_ratio, posting_efficiency,
        log_followers, log_views,
    ]]))

    probabilities   = rf.predict_proba(input_scaled)[0]
    prediction      = int(rf.predict(input_scaled)[0])

    # Apply tuned threshold for Fake Influencer
    if probabilities[fake_idx] >= fake_thresh:
        prediction = fake_idx

    predicted_label = class_names[prediction]
    confidence      = max(probabilities) * 100

    # ── Result ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### Result")
    st.markdown(f"**User Type:** {predicted_label}")
    st.markdown(f"**Status:** {STATUS[predicted_label]}")
    st.markdown(f"**Confidence:** {confidence:.1f}%")
    st.markdown(f"**Engagement Rate:** {engagement_rate:.2f}%")
    st.markdown(f"**Views Ratio:** {likes_views_ratio:.3f}")