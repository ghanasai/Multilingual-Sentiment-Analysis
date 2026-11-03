import pandas as pd
from pathlib import Path
from lingua import Language, LanguageDetectorBuilder

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT / "data" / "raw" / "training_dataset"
    / "Multilingual_sentiment_analysis.csv"
)

OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading training dataset...")

df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=["Sentence", "Label"]).copy()

df["Sentence"] = df["Sentence"].astype(str).str.strip()
df["Label"] = df["Label"].astype(str).str.strip().str.lower()
df = df[df["Sentence"] != ""].copy()

# Lingua is used only for English, Hindi,
# Marathi and other Devanagari/Latin languages.
detector = (
    LanguageDetectorBuilder
    .from_languages(
        Language.ENGLISH,
        Language.HINDI,
        Language.MARATHI
    )
    .build()
)

def identify_language(sentence):
    counts = {
        "te": 0,
        "ta": 0,
        "ml": 0,
        "devanagari": 0,
        "latin": 0,
        "other": 0
    }

    for char in sentence:
        code = ord(char)

        if 0x0C00 <= code <= 0x0C7F:
            counts["te"] += 1
        elif 0x0B80 <= code <= 0x0BFF:
            counts["ta"] += 1
        elif 0x0D00 <= code <= 0x0D7F:
            counts["ml"] += 1
        elif 0x0900 <= code <= 0x097F:
            counts["devanagari"] += 1
        elif char.isascii() and char.isalpha():
            counts["latin"] += 1
        elif char.isalpha():
            counts["other"] += 1

    dominant = max(counts, key=counts.get)

    if counts[dominant] == 0:
        return "other"

    if dominant in ["te", "ta", "ml"]:
        return dominant

    if dominant == "other":
        return "other"

    if dominant == "devanagari":
        prediction = detector.detect_language_of(sentence)

        if prediction == Language.HINDI:
            return "hi"

        return "other"

    if dominant == "latin":
        prediction = detector.detect_language_of(sentence)

        if prediction == Language.ENGLISH:
            return "en"

        return "other"

    return "other"


print("Detecting languages...")
print("Please wait. Processing 131,018 records.")

df["language"] = [
    identify_language(sentence)
    for sentence in df["Sentence"]
]

print("\nDetected language distribution:")
print(df["language"].value_counts())

target_df = df[
    df["language"].isin(["en", "hi", "te", "ta", "ml"])
].copy()

print("\nTarget language sentiment distribution:")
print(pd.crosstab(
    target_df["language"],
    target_df["Label"]
))

output_file = (
    OUTPUT_DIR / "phase1_language_predictions.csv"
)

target_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:", output_file)
print("\nPHASE 1.7 COMPLETED SUCCESSFULLY!")