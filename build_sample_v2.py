import pandas as pd

INPUT_PATH = r"C:\Users\santhiya\Downloads\archive (3)\Review_db.csv"
OUTPUT_PATH = r"C:\Users\santhiya\Downloads\archive (3)\top_places_reviews.csv"

TOP_N_PLACES = 80
MAX_REVIEWS_PER_PLACE = 300

print("Step 1: Loading full dataset...")
df = pd.read_csv(INPUT_PATH)
df = df.drop(columns=['Date'])
print("Loaded. Total rows:", df.shape[0])

print("Step 2: Counting reviews per place...")
place_counts = df['Place'].value_counts()
print("Top 15 most-reviewed places:")
print(place_counts.head(15))

print("Step 3: Filtering to top places...")
top_places = place_counts.head(TOP_N_PLACES).index
filtered = df[df['Place'].isin(top_places)].copy()
print("Rows after filtering:", filtered.shape[0])

print("Step 4: Sampling up to 300 reviews per place...")
filtered_shuffled = filtered.sample(frac=1, random_state=42).reset_index(drop=True)
capped = filtered_shuffled.groupby('Place', sort=False).head(MAX_REVIEWS_PER_PLACE)
capped = capped.reset_index(drop=True)
print("Rows after capping:", capped.shape[0])

print("Step 5: Final stats...")
print("Unique places:", capped['Place'].nunique())
print("Unique cities:", capped['City'].nunique())

print("Step 6: Saving to CSV...")
capped.to_csv(OUTPUT_PATH, index=False)
print("DONE. Saved to:", OUTPUT_PATH)