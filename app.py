from pathlib import Path
import streamlit as st
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

BASELINE_PATH = ROOT / "models" / "baseline_logistic_regression.joblib"
MBERT_PATH = ROOT / "models" / "mbert_full_dataset"

st.set_page_config(
    page_title="Multilingual Sentiment Analysis",
    page_icon="🌐",
    layout="wide"
)

# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 8px;
}
.subtitle {
    text-align: center;
    color: #999;
    margin-bottom: 30px;
}
.result-positive {
    background: #17633c;
    padding: 16px;
    border-radius: 10px;
    color: white;
    font-weight: bold;
}
.result-negative {
    background: #913b3b;
    padding: 16px;
    border-radius: 10px;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_baseline():
    return joblib.load(BASELINE_PATH)


@st.cache_resource
def load_mbert():
    tokenizer = AutoTokenizer.from_pretrained(
        str(MBERT_PATH),
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MBERT_PATH),
        local_files_only=True
    )

    model.to("cpu")
    model.eval()

    return tokenizer, model


# --------------------------------------------------
# PREDICTION FUNCTIONS
# --------------------------------------------------

def predict_baseline(text):
    model = load_baseline()

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    classes = list(model.classes_)
    confidence = float(
        probabilities[classes.index(prediction)]
    )

    label = str(prediction).lower()

    if label in ("1", "positive"):
        label = "positive"
    elif label in ("0", "negative"):
        label = "negative"

    return label, confidence


def predict_mbert(text):
    tokenizer, model = load_mbert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    prediction_id = int(torch.argmax(probabilities).item())

    label = model.config.id2label[prediction_id].lower()
    confidence = float(probabilities[prediction_id].item())

    return label, confidence


# --------------------------------------------------
# LANGUAGE SAMPLES
# --------------------------------------------------

SAMPLES = {
    "English": "This movie is very good.",
    "Hindi": "यह फिल्म बहुत अच्छी है।",
    "Telugu": "ఈ సినిమా చాలా బాగుంది.",
    "Tamil": "இந்த படம் மிகவும் நன்றாக உள்ளது.",
    "Malayalam": "ഈ സിനിമ വളരെ നല്ലതാണ്."
}

# --------------------------------------------------
# DISPLAY RESULT
# --------------------------------------------------

def show_result(model_name, label, confidence):
    st.subheader(model_name)

    if label == "positive":
        st.markdown(
            '<div class="result-positive">'
            '😊 Positive Sentiment'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-negative">'
            '😞 Negative Sentiment'
            '</div>',
            unsafe_allow_html=True
        )

    st.metric(
        "Model Confidence",
        f"{confidence * 100:.2f}%"
    )


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '🌐 Multilingual Sentiment Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Sentiment Classification Using Machine Learning '
    'and Full-Dataset Multilingual BERT'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

left, right = st.columns([1, 2])

with left:
    language = st.selectbox(
        "Select Language",
        list(SAMPLES.keys()),
        index=2
    )

    model_choice = st.radio(
        "Select Model",
        [
            "Logistic Regression",
            "Full-Dataset mBERT",
            "Compare Both Models"
        ],
        index=2
    )

with right:
    text = st.text_area(
        "Enter Your Review",
        value=SAMPLES[language],
        height=140
    )

    analyze = st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

if analyze:
    if not text.strip():
        st.warning("Please enter a review.")
    else:
        st.divider()
        st.header("Prediction Results")
        st.caption(f"Selected Language: {language}")

        if model_choice == "Logistic Regression":
            try:
                label, confidence = predict_baseline(text)
                show_result(
                    "Logistic Regression",
                    label,
                    confidence
                )
            except Exception as error:
                st.error(f"Baseline model error: {error}")

        elif model_choice == "Full-Dataset mBERT":
            try:
                label, confidence = predict_mbert(text)
                show_result(
                    "Full-Dataset mBERT",
                    label,
                    confidence
                )
            except Exception as error:
                st.error(f"mBERT model error: {error}")

        else:
            col1, col2 = st.columns(2)

            baseline_result = None
            mbert_result = None

            with col1:
                try:
                    baseline_result = predict_baseline(text)
                    show_result(
                        "Logistic Regression",
                        *baseline_result
                    )
                except Exception as error:
                    st.error(f"Baseline error: {error}")

            with col2:
                try:
                    mbert_result = predict_mbert(text)
                    show_result(
                        "Full-Dataset mBERT",
                        *mbert_result
                    )
                except Exception as error:
                    st.error(f"mBERT error: {error}")

            if baseline_result and mbert_result:
                if baseline_result[0] == mbert_result[0]:
                    st.success(
                        "Both models predict the same sentiment."
                    )
                else:
                    st.warning(
                        "The models predict different sentiments."
                    )

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()
st.header("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Logistic Regression Accuracy",
        "81.57%"
    )

with col2:
    st.metric(
        "Previous CPU mBERT Accuracy",
        "74.52%"
    )

with col3:
    st.metric(
        "Full-Dataset mBERT",
        "Evaluation Pending"
    )

st.caption(
    "Accuracy values are from the 624-review validation "
    "dataset. The final full-dataset model accuracy must "
    "be measured separately. Model confidence is not "
    "a guarantee of correctness."
)

st.divider()

st.markdown(
    "**Supported Languages:** "
    "English | Hindi | Telugu | Tamil | Malayalam"
)

st.caption(
    "Multilingual Sentiment Analysis Project | "
    "TF-IDF + Logistic Regression vs Multilingual BERT"
)