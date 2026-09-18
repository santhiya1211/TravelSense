 
import pandas as pd
 
INPUT_PATH = r"C:\Users\santhiya\Downloads\archive (3)\top_places_with_sentiment.csv"
OUTPUT_REVIEWS_PATH = r"C:\Users\santhiya\Downloads\archive (3)\reviews_with_aspects.csv"
OUTPUT_PROFILE_PATH = r"C:\Users\santhiya\Downloads\archive (3)\destination_aspect_profile.csv"
 
print("Loading sentiment-scored reviews...")
df = pd.read_csv(INPUT_PATH)
print("Loaded:", df.shape)
 
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
 
def find_aspects(clean_text):
    if not isinstance(clean_text, str):
        return []
    tokens = set(clean_text.split())
    found = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if tokens.intersection(keywords):
            found.append(aspect)
    return found
 
print("Tagging reviews with aspects...")
df['aspects_mentioned'] = df['clean_text'].apply(find_aspects)
df['num_aspects'] = df['aspects_mentioned'].apply(len)
 
print("\nHow many reviews mention at least one aspect:",
      (df['num_aspects'] > 0).sum(), "out of", len(df))
 
# ---------------------------------------------------------
# 3. EXPLODE INTO ONE ROW PER (REVIEW, ASPECT) FOR AGGREGATION
# ---------------------------------------------------------
exploded = df.explode('aspects_mentioned').dropna(subset=['aspects_mentioned'])
print("\nAspect mention counts overall:")
print(exploded['aspects_mentioned'].value_counts())
 
profile = (
    exploded.groupby(['Place', 'aspects_mentioned'])['vader_compound']
    .mean()
    .reset_index()
    .rename(columns={'vader_compound': 'avg_sentiment'})
)
 
# Pivot so each aspect becomes its own column
profile_wide = profile.pivot(index='Place', columns='aspects_mentioned', values='avg_sentiment')
profile_wide.columns.name = None
profile_wide = profile_wide.reset_index()
 
# Add overall destination stats: avg rating, avg overall sentiment, review count, city
overall_stats = df.groupby('Place').agg(
    City=('City', 'first'),
    review_count=('Review', 'count'),
    avg_rating=('Rating', 'mean'),
    avg_overall_sentiment=('vader_compound', 'mean')
).reset_index()
 
destination_profile = overall_stats.merge(profile_wide, on='Place', how='left')
 
print("\n--- Destination Aspect Profile (sample) ---")
print(destination_profile.head(10).to_string())
 
# ---------------------------------------------------------
# 5. SAVE OUTPUTS
# ---------------------------------------------------------
df.to_csv(OUTPUT_REVIEWS_PATH, index=False)
destination_profile.to_csv(OUTPUT_PROFILE_PATH, index=False)
 
print(f"\nSaved review-level aspects to: {OUTPUT_REVIEWS_PATH}")
print(f"Saved destination aspect profile to: {OUTPUT_PROFILE_PATH}")
 
