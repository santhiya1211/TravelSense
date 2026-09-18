"""
TravelSense - Streamlit Dashboard (Full Coverage Version)
-------------------------------------------------------------
Run with: streamlit run app.py

Uses the FULL dataset (all ~14,494 places), with a confidence tier
(High/Medium/Low) shown for every destination based on review count,
so users know how reliable each score is.

Requires these files (update paths below):
- all_reviews_processed.csv
- all_destinations_profile.csv
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# CONFIG - update these paths to match your machine
# ---------------------------------------------------------
REVIEWS_PATH = r"C:\Users\santhiya\Downloads\archive (3)\all_reviews_processed.csv"
PROFILE_PATH = r"C:\Users\santhiya\Downloads\archive (3)\all_destinations_profile.csv"
COORDS_PATH = r"C:\Users\santhiya\Downloads\archive (3)\city_coordinates.csv"

ASPECT_COLUMNS = [
    "cleanliness", "staff_service", "price_value", "food",
    "crowd", "accessibility", "scenery_experience", "safety"
]

CONFIDENCE_COLORS = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}

st.set_page_config(page_title="TravelSense", layout="wide")

# ---------------------------------------------------------
# LOAD DATA (cached so it doesn't reload on every interaction)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    reviews = pd.read_csv(REVIEWS_PATH)
    profile = pd.read_csv(PROFILE_PATH)
    # Fill missing aspect scores with the overall column average
    # (this happens when a destination has zero reviews mentioning that aspect)
    for col in ASPECT_COLUMNS:
        if col in profile.columns:
            profile[col] = profile[col].fillna(profile[col].mean())
    # Friendly confidence label with emoji
    profile['confidence_display'] = profile['confidence'].map(CONFIDENCE_COLORS) + " " + profile['confidence']

    # Merge in city coordinates for the map view, if the file exists
    try:
        coords = pd.read_csv(COORDS_PATH)
        profile = profile.merge(coords, on='City', how='left')
        # Add a small random jitter so multiple destinations in the same
        # city don't render as a single overlapping dot on the map
        rng = np.random.default_rng(42)
        n = len(profile)
        profile['map_lat'] = profile['lat'] + rng.uniform(-0.03, 0.03, n)
        profile['map_lon'] = profile['lon'] + rng.uniform(-0.03, 0.03, n)
    except FileNotFoundError:
        profile['lat'] = None
        profile['lon'] = None
        profile['map_lat'] = None
        profile['map_lon'] = None

    return reviews, profile

reviews, profile = load_data()

st.title("🧭 TravelSense")
st.caption("NLP-Based Analysis of Tourist Reviews and Destination Experiences")
st.caption(f"Covering all {profile['Place'].nunique():,} destinations in the dataset, "
           f"each tagged with a data confidence level based on review volume.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Find a Destination", "🔁 Similar Destinations", "🗺️ Map View"])

# ---------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------
with tab1:
    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{len(reviews):,}")
    col2.metric("Destinations", f"{profile['Place'].nunique():,}")
    col3.metric("Avg Sentiment (VADER)", f"{reviews['vader_compound'].mean():.2f}")
    high_conf_count = (profile['confidence'] == 'High').sum()
    col4.metric("High-Confidence Destinations", f"{high_conf_count:,}")

    st.markdown("### Confidence Tier Breakdown")
    st.caption("How many reviews back each destination's score — more reviews means a more reliable score.")
    conf_counts = profile['confidence'].value_counts().reindex(['High', 'Medium', 'Low'])
    fig0, ax0 = plt.subplots(figsize=(5, 3))
    ax0.bar(conf_counts.index, conf_counts.values, color=['#4CAF50', '#FFC107', '#F44336'])
    ax0.set_ylabel("Number of Destinations")
    st.pyplot(fig0)
    st.caption("🟢 High = 100+ reviews  |  🟡 Medium = 20-99 reviews  |  🔴 Low = under 20 reviews")

    st.markdown("### Sentiment Distribution (all reviews)")
    fig, ax = plt.subplots(figsize=(5, 3))
    reviews['vader_label'].value_counts().plot(kind='bar', ax=ax, color=['#4CAF50', '#F44336', '#FFC107'])
    ax.set_ylabel("Number of Reviews")
    st.pyplot(fig)

    st.markdown("### Top 10 Destinations by Average Sentiment (High confidence only)")
    top_sent = (
        profile[profile['confidence'] == 'High']
        .sort_values('avg_overall_sentiment', ascending=False)
        .head(10)
    )
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.barh(top_sent['Place'], top_sent['avg_overall_sentiment'], color='#2196F3')
    ax2.invert_yaxis()
    ax2.set_xlabel("Average Sentiment Score")
    st.pyplot(fig2)

    st.markdown("### Average Sentiment by Aspect (overall, all destinations)")
    aspect_avgs = profile[ASPECT_COLUMNS].mean().sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.bar(aspect_avgs.index, aspect_avgs.values, color='#9C27B0')
    plt.xticks(rotation=45, ha='right')
    ax3.set_ylabel("Average Sentiment")
    st.pyplot(fig3)

# ---------------------------------------------------------
# TAB 2: PREFERENCE-BASED RECOMMENDER
# ---------------------------------------------------------
with tab2:
    st.subheader("Find a Destination Based on What You Care About")

    selected_aspects = st.multiselect(
        "Select the aspects that matter most to you:",
        options=ASPECT_COLUMNS,
        default=["scenery_experience", "food"]
    )

    top_n = st.slider("Number of recommendations", 3, 15, 5)
    confidence_filter = st.multiselect(
        "Only show destinations with this data confidence:",
        options=["High", "Medium", "Low"],
        default=["High", "Medium"]
    )
    st.caption("Tip: keep 'Low' unchecked unless you want to explore lesser-known places with limited review data.")

    if selected_aspects and confidence_filter:
        candidates = profile[profile['confidence'].isin(confidence_filter)].copy()
        candidates['match_score'] = candidates[selected_aspects].mean(axis=1)
        result = candidates.sort_values('match_score', ascending=False).head(top_n)

        st.markdown(f"### Top {top_n} Destinations for: {', '.join(selected_aspects)}")
        display_cols = ['Place', 'City', 'avg_rating', 'review_count', 'confidence_display', 'match_score'] + selected_aspects
        st.dataframe(
            result[display_cols].rename(columns={'confidence_display': 'confidence'}).reset_index(drop=True),
            use_container_width=True
        )
    else:
        st.info("Select at least one aspect and one confidence level above to get recommendations.")

# ---------------------------------------------------------
# TAB 3: SIMILARITY-BASED RECOMMENDER
# ---------------------------------------------------------
with tab3:
    st.subheader("Find Destinations Similar to One You Liked")

    conf_filter_sim = st.multiselect(
        "Only search within destinations with this data confidence:",
        options=["High", "Medium", "Low"],
        default=["High", "Medium"],
        key="conf_sim"
    )
    searchable = profile[profile['confidence'].isin(conf_filter_sim)] if conf_filter_sim else profile

    place_choice = st.selectbox("Pick a destination you liked:", sorted(searchable['Place'].unique()))
    top_n_sim = st.slider("Number of similar destinations to show", 3, 15, 5, key="sim_slider")

    vectors = profile[ASPECT_COLUMNS].values
    sim_matrix_row_idx = profile.index[profile['Place'] == place_choice][0]

    # Compute similarity of the chosen place against the searchable subset only
    from numpy import dot
    from numpy.linalg import norm

    chosen_vector = profile.loc[sim_matrix_row_idx, ASPECT_COLUMNS].values.reshape(1, -1)
    searchable_vectors = searchable[ASPECT_COLUMNS].values
    sims = cosine_similarity(chosen_vector, searchable_vectors)[0]

    searchable = searchable.copy()
    searchable['similarity'] = sims
    searchable = searchable[searchable['Place'] != place_choice]
    top_similar = searchable.sort_values('similarity', ascending=False).head(top_n_sim)

    # --- Show the chosen destination's own aspect profile first ---
    chosen_row = profile[profile['Place'] == place_choice][
        ['Place', 'City', 'avg_rating', 'review_count', 'confidence_display'] + ASPECT_COLUMNS
    ]
    st.markdown(f"### {place_choice} — Aspect Scores")
    st.dataframe(
        chosen_row.rename(columns={'confidence_display': 'confidence'}).reset_index(drop=True),
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(7, 3))
    chosen_scores = chosen_row[ASPECT_COLUMNS].iloc[0]
    ax.bar(chosen_scores.index, chosen_scores.values, color='#FF9800')
    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel("Sentiment Score")
    ax.set_title(f"{place_choice}: sentiment by aspect")
    st.pyplot(fig)

    # --- Then show similar destinations, with their full aspect scores too ---
    st.markdown(f"### Destinations Similar to **{place_choice}**")
    st.caption("Ranked by overall similarity across all aspects — aspect scores included for direct comparison.")
    display_cols_sim = ['Place', 'City', 'avg_rating', 'review_count', 'confidence_display', 'similarity'] + ASPECT_COLUMNS
    st.dataframe(
        top_similar[display_cols_sim].rename(columns={'confidence_display': 'confidence'}).reset_index(drop=True),
        use_container_width=True
    )

# ---------------------------------------------------------
# TAB 4: MAP VIEW
# ---------------------------------------------------------
with tab4:
    st.subheader("Destinations on the Map, Colored by Sentiment")

    if profile['lat'].isna().all():
        st.warning(
            "City coordinates haven't been generated yet. Run `geocode_cities.py` once "
            "to create `city_coordinates.csv`, then reload this app to see the map."
        )
    else:
        map_confidence = st.multiselect(
            "Show destinations with confidence:",
            options=["High", "Medium", "Low"],
            default=["High", "Medium"],
            key="map_conf"
        )

        map_data = profile[profile['confidence'].isin(map_confidence)].dropna(subset=['map_lat', 'map_lon'])

        st.caption(
            f"Showing {len(map_data):,} of {profile['Place'].nunique():,} destinations "
            f"({profile['lat'].notna().sum():,} have known coordinates)."
        )

        if len(map_data) == 0:
            st.info("No destinations match the selected confidence levels.")
        else:
            color_min = map_data['avg_overall_sentiment'].quantile(0.02)
            color_max = map_data['avg_overall_sentiment'].quantile(0.98)

            map_kwargs = dict(
                lat="map_lat",
                lon="map_lon",
                color="avg_overall_sentiment",
                size="review_count",
                size_max=18,
                hover_name="Place",
                hover_data={
                    "City": True,
                    "avg_rating": ":.2f",
                    "avg_overall_sentiment": ":.2f",
                    "review_count": True,
                    "confidence": True,
                    "map_lat": False,
                    "map_lon": False,
                },
                color_continuous_scale="RdYlGn",
                range_color=[color_min, color_max],
                zoom=3.7,
                center={"lat": 22.5, "lon": 80.0},
                height=650,
            )
            if hasattr(px, "scatter_map"):
                fig = px.scatter_map(map_data, map_style="open-street-map", **map_kwargs)
            else:
                fig = px.scatter_mapbox(map_data, mapbox_style="open-street-map", **map_kwargs)

            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                f"🔴 Red = relatively lower sentiment · 🟢 Green = relatively higher sentiment "
                f"(scale stretched to this view's actual range: {color_min:.2f} to {color_max:.2f}) · "
                "Bubble size = number of reviews. Positions are jittered slightly "
                "within each city so overlapping destinations remain visible."
            )