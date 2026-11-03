import pandas as pd
import re
import unicodedata
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"

train = pd.read_csv(
    RESULTS / "phase1_screened_candidates.csv"
)

val = pd.read_csv(
    PROCESSED / "validation_clean.csv"
)

test = pd.read_csv(
    PROCESSED / "test_clean.csv"
)

def normalize(text):
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", text.casefold()).strip()

print("=" * 60)
print("PHASE 1.9 - FINAL TRAINING AUDIT")
print("=" * 60)

# Audit duplicate text with conflicting labels
raw = pd.read_csv(
    RESULTS / "phase1_language_predictions.csv"
)

raw["Label"] = raw["Label"].astype(str).str.lower()
raw = raw[raw["Label"].isin(["positive", "negative"])].copy()
raw["norm"] = raw["Sentence"].apply(normalize)

conflicts = (
    raw.groupby("norm")["Label"]
    .nunique()
)

conflict_texts = set(conflicts[conflicts > 1].index)

print("\nConflicting-label sentences:",
      len(conflict_texts))

# Exclude conflicts from screened candidates
train["norm"] = train["Sentence"].apply(normalize)
train = train[~train["norm"].isin(conflict_texts)].copy()

# Near-duplicate comparison within the same script/language.
# This is a screening heuristic, not proof of independence.
evaluation = pd.concat([val, test], ignore_index=True)

threshold = 0.85
flagged_rows = []

for language in ["en", "hi", "te", "ta", "ml"]:
    candidate = train[train["language"] == language].copy()

    if language == "en":
        references = evaluation["english_text"].dropna()
    else:
        references = evaluation[
            evaluation["language"] == language
        ]["text"].dropna()

    reference_texts = (
        references.apply(normalize)
        .drop_duplicates()
        .tolist()
    )

    if candidate.empty or not reference_texts:
        continue

    candidate_texts = candidate["norm"].tolist()

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
        max_features=100000
    )

    matrix = vectorizer.fit_transform(
        reference_texts + candidate_texts
    )

    reference_matrix = matrix[:len(reference_texts)]
    candidate_matrix = matrix[len(reference_texts):]

    neighbors = NearestNeighbors(
        n_neighbors=1,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1
    )

    neighbors.fit(reference_matrix)

    distances, indices = neighbors.kneighbors(
        candidate_matrix
    )

    similarities = 1 - distances.ravel()

    candidate["max_similarity"] = similarities
    candidate["closest_eval_text"] = [
        reference_texts[i] for i in indices.ravel()
    ]

    flagged = candidate[
        candidate["max_similarity"] >= threshold
    ].copy()

    print(
        f"{language}: {len(flagged)} near-duplicate "
        f"candidates (similarity >= {threshold})"
    )

    flagged_rows.append(flagged)

if flagged_rows:
    flagged_df = pd.concat(flagged_rows, ignore_index=True)
else:
    flagged_df = pd.DataFrame()

flagged_df.to_csv(
    RESULTS / "phase1_near_duplicate_review.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nRemaining after conflict exclusion:",
      len(train))

print("\nSaved near-duplicate review file.")
print("PHASE 1.9 AUDIT COMPLETED!")