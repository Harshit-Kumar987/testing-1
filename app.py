# app.py — Instagram Influencer Detection
# Deploy: streamlit.io

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Influencer Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Artifacts ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    rf       = joblib.load('model/rf_model.pkl')
    scaler   = joblib.load('model/scaler.pkl')
    features = joblib.load('model/feature_names.pkl')
    cat_map  = joblib.load('model/category_mapping.pkl')
    return rf, scaler, features, cat_map

@st.cache_data
def load_data():
    df = pd.read_csv('data/influencers_scored.csv')
    # Remove adult content
    df = df[~df['category'].str.lower().str.contains(
        'adult', na=False
    )]
    return df

rf, scaler, FEATURES, cat_map = load_artifacts()
df = load_data()

# Remove adult from category mapping
cat_map = {k: v for k, v in cat_map.items()
           if 'adult' not in k.lower()}

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("Filters")

tiers = st.sidebar.multiselect(
    "Influencer Tier",
    options=['Mega', 'Macro', 'Micro', 'Non-Influencer'],
    default=['Mega', 'Macro', 'Micro']
)

min_score = st.sidebar.slider(
    "Minimum Influencer Score", 0, 100, 20
)

countries = ['All'] + sorted(
    df['audience_country'].dropna().unique().tolist()
)
country = st.sidebar.selectbox("Audience Country", countries)

categories = ['All'] + sorted(
    df['category'].dropna().unique().tolist()
)
category = st.sidebar.selectbox("Category", categories)

top_n = st.sidebar.slider("Show Top N", 5, 200, 20)

# ── Apply Filters ──────────────────────────────────────────────
filtered = df[
    df['tier'].isin(tiers) &
    (df['influencer_score'] >= min_score)
].copy()

if country != 'All':
    filtered = filtered[filtered['audience_country'] == country]
if category != 'All':
    filtered = filtered[filtered['category'] == category]

filtered = filtered.sort_values(
    'influencer_score', ascending=False
).head(top_n)

# ── Header ─────────────────────────────────────────────────────
st.title("Instagram Influencer Detection")
st.divider()

# ── KPI Cards ──────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Results",       len(filtered))
k2.metric("Avg Score",     f"{filtered['influencer_score'].mean():.1f}")
k3.metric("Avg Eng. Rate", f"{filtered['engagement_rate'].mean():.3f}%")
k4.metric("Avg Followers", f"{filtered['followers'].mean()/1e6:.2f}M")
k5.metric("Categories",    filtered['category'].nunique())
st.divider()

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🏆 Leaderboard",
    "📊 Analytics",
    "🔍 Predict"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — LEADERBOARD
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Influencer Rankings")

    display_cols = ['rank', 'name', 'category', 'audience_country',
                    'followers', 'engagement_rate',
                    'influencer_score', 'tier']
    display_cols = [c for c in display_cols if c in filtered.columns]

    def color_tier(val):
        colors = {
            'Mega'          : 'background-color: #1565C0; color: white',
            'Macro'         : 'background-color: #1976D2; color: white',
            'Micro'         : 'background-color: #42A5F5; color: white',
            'Non-Influencer': 'background-color: #EEEEEE; color: #616161'
        }
        return colors.get(val, '')

    styled = (
        filtered[display_cols]
        .reset_index(drop=True)
        .style
        .map(color_tier, subset=['tier'])
        .format({
            'followers'       : '{:,.0f}',
            'engagement_rate' : '{:.3f}%',
            'influencer_score': '{:.1f}'
        })
    )
    st.dataframe(styled, use_container_width=True)

    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇ Download Results as CSV",
        data=csv,
        file_name='influencers.csv',
        mime='text/csv'
    )

# ══════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Tier Distribution**")
        
        # Remove Nano from the chart data
        tier_counts = df[df['tier'] != 'Nano']['tier'].value_counts()
        
        tier_colors = {
            'Mega'          : '#0D47A1',
            'Macro'         : '#1976D2',
            'Micro'         : '#42A5F5',
            'Non-Influencer': "#76797B"
        }
        colors = [tier_colors.get(t, '#90CAF9') for t in tier_counts.index]

        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            tier_counts,
            labels=None,           # remove labels from pie itself
            autopct='%1.1f%%',
            colors=colors,
            startangle=140,
            pctdistance=0.75
        )
        # Add clean legend instead of overlapping labels
        ax.legend(
            wedges,
            tier_counts.index,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            fontsize=9
        )
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color('white')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Top 10 Audience Countries**")
        top_countries = df['audience_country'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(5, 4))
        top_countries.plot(kind='barh', ax=ax,
                           color='#42A5F5', edgecolor='white')
        ax.invert_yaxis()
        ax.set_xlabel('Count')
        st.pyplot(fig)
        plt.close()

    st.markdown("**Top Categories by Avg Influencer Score**")
    cat_scores = (
        df.groupby('category')['influencer_score']
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    cat_scores.plot(kind='bar', ax=ax,
                    color='#1565C0', edgecolor='white')
    ax.set_xlabel('Category')
    ax.set_ylabel('Avg Influencer Score')
    ax.set_title('Top Categories by Avg Score')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Engagement Rate vs Followers**")
    fig, ax = plt.subplots(figsize=(9, 4))
    sc = ax.scatter(
        np.log1p(df['followers']),
        df['engagement_rate'],
        c=df['influencer_score'],
        cmap='Blues', alpha=0.7, s=30
    )
    plt.colorbar(sc, ax=ax, label='Influencer Score')
    ax.set_xlabel('Log(Followers)')
    ax.set_ylabel('Engagement Rate (%)')
    ax.set_title('Reach vs Engagement (colored by Score)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 3 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Predict — New Instagram Account")
    st.markdown("Enter account details to get the influencer score.")

    c1, c2, c3 = st.columns(3)

    with c1:
        inp_rank      = st.number_input(
            "World Rank", 1, 10000, value=250)
        inp_followers = st.number_input(
            "Followers", 1000, 500_000_000,
            value=1_000_000, step=100_000)

    with c2:
        inp_auth_eng = st.number_input(
            "Authentic Engagement", 0, 50_000_000,
            value=50_000, step=1000)
        inp_eng_avg  = st.number_input(
            "Engagement Avg.", 0, 50_000_000,
            value=45_000, step=1000)

    with c3:
        inp_category = st.selectbox(
            "Category",
            options=sorted(list(cat_map.keys()))
        )

    if st.button("Get Influencer Score", type="primary"):
        eng_rate    = (inp_auth_eng / (inp_followers + 1)) * 100
        reach       = np.log1p(inp_followers)
        rank_s      = 1.0 / inp_rank
        consistency = min(inp_eng_avg / (inp_auth_eng + 1), 5.0)
        cat_enc     = cat_map.get(inp_category, 0)

        input_arr    = np.array([[eng_rate, reach, rank_s,
                                   consistency, cat_enc]])
        input_scaled = scaler.transform(input_arr)

        prob  = rf.predict_proba(input_scaled)[0][1]
        score = round(prob * 100, 2)

        tier = ('Mega'           if score >= 80 else
                'Macro'          if score >= 60 else
                'Micro'          if score >= 40 else
                'Non-Influencer')

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Influencer Score", f"{score} / 100")
        r2.metric("Tier",             tier)
        r3.metric("Engagement Rate",  f"{eng_rate:.3f}%")
        st.progress(int(score))