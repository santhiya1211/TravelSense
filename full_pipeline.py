"""
TravelSense - FULL PIPELINE (all places, with confidence tiers)
--------------------------------------------------------------------
Processes the ENTIRE dataset (all ~14,494 places, ~1.48M reviews),
not just the top 80. To keep this fast at this scale:

  - Sentiment: VADER only (fast, works directly on raw text,
    no tokenizing/lemmatizing needed). TextBlob was already
    validated for agreement on the smaller sample earlier.
  - Aspects: detected via simple keyword search on the already-
    cleaned 'Review' column (no per-review tokenize/lemmatize).

Each destination gets a CONFIDENCE TIER based on how many reviews
it has, so the dashboard can warn users when a score is based on
very few reviews instead of silently treating it as equally
reliable as a place with hundreds of reviews.

Expect this to take several minutes on the full dataset - that's
normal given the size. Progress is printed so you can see it's
working.
"""

import pandas as pd
import re
from nltk.sentiment import SentimentIntensityAnalyzer

INPUT_PATH = r"C:\Users\santhiya\Downloads\archive (3)\Review_db.csv"
OUTPUT_REVIEWS_PATH = r"C:\Users\santhiya\Downloads\archive (3)\all_reviews_processed.csv"
OUTPUT_PROFILE_PATH = r"C:\Users\santhiya\Downloads\archive (3)\all_destinations_profile.csv"

ASPECT_KEYWORDS = {
    "cleanliness": ["clean", "dirty", "hygiene", "hygienic", "spotless", "trash",
                     "garbage", "litter", "maintained", "filthy", "unclean"],
    "staff_service": ["staff", "service", "guide", "helpful", "rude", "friendly",
                        "behaviour", "behavior", "employee", "manager", "cooperative"],
    "price_value": ["price", "expensive", "cheap", "cost", "worth", "value",
                      "overpriced", "affordable", "fee", "ticket", "money"],
    "food": ["food", "restaurant", "meal", "breakfast", "lunch", "dinner",
              "taste", "delicious", "snack", "cuisine", "eat"],
    "crowd": ["crowd", "crowded", "rush", "queue", "line", "busy", "peaceful",
               "quiet", "packed", "tourist"],
    "accessibility": ["parking", "transport", "reach", "access", "distance",
                        "located", "location", "connectivity", "far", "near"],
    "scenery_experience": ["beautiful", "view", "scenery", "amazing", "stunning",
                             "architecture", "history", "historical", "peaceful",
                             "experience", "atmosphere"],
    "safety": ["safe", "safety", "unsafe", "security", "secure", "danger"]
}

def confidence_tier(count):
    if count >= 100:
        return "High"
    elif count >= 20:
        return "Medium"
    else:
        return "Low"

print("Step 1: Loading full dataset (this is the slow part, ~30-60s)...")
df = pd.read_csv(INPUT_PATH)
df = df.drop(columns=['Date'], errors='ignore')
print("Loaded:", df.shape)
print("Unique places:", df['Place'].nunique())

print("\nStep 2: Running VADER sentiment on all reviews (this may take a few minutes)...")
sia = SentimentIntensityAnalyzer()

def vader_compound(text):
    return sia.polarity_scores(str(text))['compound']

df['vader_compound'] = df['Raw_Review'].apply(vader_compound)
df['vader_label'] = df['vader_compound'].apply(
    lambda c: 'positive' if c >= 0.05 else ('negative' if c <= -0.05 else 'neutral')
)
print("Sentiment done. Sample distribution:")
print(df['vader_label'].value_counts())

print("\nStep 3: Detecting aspects mentioned in each review...")
def find_aspects(text):
    if not isinstance(text, str):
        return []
    words = set(text.split())
    return [aspect for aspect, kws in ASPECT_KEYWORDS.items() if words.intersection(kws)]

df['aspects_mentioned'] = df['Review'].apply(find_aspects)
print("Aspect detection done.")

print("\nStep 4: Building per-destination profile with confidence tiers...")
exploded = df.explode('aspects_mentioned').dropna(subset=['aspects_mentioned'])
aspect_avg = (
    exploded.groupby(['Place', 'aspects_mentioned'])['vader_compound']
    .mean()
    .reset_index()
    .pivot(index='Place', columns='aspects_mentioned', values='vader_compound')
)
aspect_avg.columns.name = None
aspect_avg = aspect_avg.reset_index()

overall_stats = df.groupby('Place').agg(
    City=('City', 'first'),
    review_count=('Review', 'count'),
    avg_rating=('Rating', 'mean'),
    avg_overall_sentiment=('vader_compound', 'mean')
).reset_index()

profile = overall_stats.merge(aspect_avg, on='Place', how='left')
profile['confidence'] = profile['review_count'].apply(confidence_tier)

print("\nConfidence tier breakdown:")
print(profile['confidence'].value_counts())

print("\nStep 5: Saving outputs...")
df.to_csv(OUTPUT_REVIEWS_PATH, index=False)
profile.to_csv(OUTPUT_PROFILE_PATH, index=False)

print(f"\nDONE.")
print(f"Saved all reviews to: {OUTPUT_REVIEWS_PATH}")
print(f"Saved full destination profile ({len(profile)} places) to: {OUTPUT_PROFILE_PATH}")