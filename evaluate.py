"""
TravelSense - Day 13: Evaluation
------------------------------------
Compares VADER sentiment labels against "ground truth" derived from
star ratings, and reports accuracy, precision, recall, F1, and a
confusion matrix. Also checks agreement with TextBlob where available.
"""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

INPUT_PATH = r"C:\Users\santhiya\Downloads\archive (3)\all_reviews_processed.csv"

print("Loading processed reviews...")
df = pd.read_csv(INPUT_PATH)
print("Loaded:", df.shape)

# ---------------------------------------------------------
# 1. DERIVE GROUND TRUTH FROM STAR RATINGS
# ---------------------------------------------------------
# Common convention: 4-5 stars = positive, 3 = neutral, 1-2 = negative
def rating_to_label(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

df['rating_label'] = df['Rating'].apply(rating_to_label)

print("\nGround truth (from ratings) distribution:")
print(df['rating_label'].value_counts())

print("\nVADER predicted distribution:")
print(df['vader_label'].value_counts())

# ---------------------------------------------------------
# 2. ACCURACY / PRECISION / RECALL / F1
# ---------------------------------------------------------
y_true = df['rating_label']
y_pred = df['vader_label']

acc = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {acc:.2%}")

print("\nClassification Report (Precision / Recall / F1 per class):")
print(classification_report(y_true, y_pred, digits=3))

print("Confusion Matrix (rows=actual, columns=predicted):")
labels = ['negative', 'neutral', 'positive']
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=[f"actual_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
print(cm_df)

# ---------------------------------------------------------
# 3. LOOK AT DISAGREEMENTS (useful for your report's limitations section)
# ---------------------------------------------------------
disagreements = df[df['rating_label'] != df['vader_label']]
print(f"\nTotal disagreements: {len(disagreements):,} out of {len(df):,} "
      f"({len(disagreements)/len(df):.1%})")

print("\nSample cases where a 5-star review was scored negative by VADER "
      "(often sarcasm, mixed sentiment, or short reviews):")
tricky = df[(df['Rating'] == 5) & (df['vader_label'] == 'negative')]
if len(tricky) > 0:
    print(tricky[['Place', 'Rating', 'vader_label', 'vader_compound', 'Raw_Review']].head(5).to_string())
else:
    print("None found in this dataset.")

print("\nSample cases where a 1-star review was scored positive by VADER:")
tricky2 = df[(df['Rating'] == 1) & (df['vader_label'] == 'positive')]
if len(tricky2) > 0:
    print(tricky2[['Place', 'Rating', 'vader_label', 'vader_compound', 'Raw_Review']].head(5).to_string())
else:
    print("None found in this dataset.")

print("\nEvaluation complete.")