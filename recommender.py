import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROFILE_PATH = r"C:\Users\santhiya\Downloads\archive (3)\destination_aspect_profile.csv"

ASPECT_COLUMNS = [
    "cleanliness", "staff_service", "price_value", "food",
    "crowd", "accessibility", "scenery_experience", "safety"
]

print("Loading destination aspect profile...")
profile = pd.read_csv(PROFILE_PATH)
print("Loaded:", profile.shape)

for col in ASPECT_COLUMNS:
    if col in profile.columns:
        profile[col] = profile[col].fillna(profile[col].mean())

def recommend_by_preference(preferred_aspects, top_n=5, min_reviews=20):
    valid_aspects = [a for a in preferred_aspects if a in ASPECT_COLUMNS]
    if not valid_aspects:
        raise ValueError(f"No valid aspects given. Choose from: {ASPECT_COLUMNS}")
    candidates = profile[profile['review_count'] >= min_reviews].copy()
    candidates['match_score'] = candidates[valid_aspects].mean(axis=1)
    result = candidates.sort_values('match_score', ascending=False).head(top_n)
    return result[['Place', 'City', 'avg_rating', 'match_score'] + valid_aspects]

def recommend_similar(place_name, top_n=5):
    if place_name not in profile['Place'].values:
        raise ValueError(f"'{place_name}' not found in destination profile.")
    vectors = profile[ASPECT_COLUMNS].values
    sim_matrix = cosine_similarity(vectors)
    idx = profile.index[profile['Place'] == place_name][0]
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, score in sim_scores[1:top_n + 1]]
    result = profile.iloc[top_indices][['Place', 'City', 'avg_rating'] + ASPECT_COLUMNS].copy()
    result['similarity'] = [score for i, score in sim_scores[1:top_n + 1]]
    return result

if __name__ == "__main__":
    print("\n=== DEMO 1: Preference-based recommendation ===")
    print("User wants: food + scenery_experience")
    recs = recommend_by_preference(["food", "scenery_experience"], top_n=5)
    print(recs.to_string())

    print("\n=== DEMO 2: Similarity-based recommendation ===")
    sample_place = profile['Place'].iloc[0]
    print(f"User liked: {sample_place}")
    similar = recommend_similar(sample_place, top_n=5)
    print(similar[['Place', 'City', 'avg_rating', 'similarity']].to_string())