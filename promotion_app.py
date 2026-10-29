# ============================================================
# HR PROMOTION PREDICTION APP
# Streamlit + CatBoost + TF-IDF + TruncatedSVD
# Compatible with train_promotion_nlp.py
# ============================================================

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier

# ============================================================
# 1. PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "promotion_dataset_.csv"
MODEL_PATH = BASE_DIR / "models" / "promotion_catboost_nlp.cbm"
METADATA_PATH = BASE_DIR / "models" / "promotion_nlp_metadata.json"
VECTORIZER_PATH = BASE_DIR / "models" / "promotion_tfidf_vectorizer.joblib"
SVD_PATH = BASE_DIR / "models" / "promotion_svd.joblib"

TEXT_COLS = [
    "Employee_Self_Review_Text",
    "Manager_Performance_Review_Text",
    "Project_Impact_Summary_Text",
    "Development_Goals_Text",
]

# ============================================================
# 2. LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    required_paths = [MODEL_PATH, METADATA_PATH, VECTORIZER_PATH, SVD_PATH]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required trained model files are missing:\n- " + "\n- ".join(missing)
            + "\n\nRun `python train_promotion_nlp.py` first."
        )

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    vectorizer = joblib.load(VECTORIZER_PATH)
    svd = joblib.load(SVD_PATH)
    return model, metadata, vectorizer, svd


@st.cache_data
def load_reference_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


def combine_text_values(values):
    """Combine the four review text fields in the same order as training."""
    return " ".join(str(values.get(col, "") or "") for col in TEXT_COLS)


def make_prediction_features(structured_values, text_values, metadata, vectorizer, svd):
    text_document = combine_text_values(text_values)
    tfidf = vectorizer.transform([text_document])
    text_components = svd.transform(tfidf)

    row = dict(structured_values)
    for i, name in enumerate(metadata["nlp_feature_names"]):
        row[name] = float(text_components[0, i])

    X_input = pd.DataFrame([row])
    feature_columns = metadata["feature_columns"]
    X_input = X_input.reindex(columns=feature_columns)

    categorical_columns = metadata["categorical_columns"]
    numeric_columns = metadata["numeric_columns"]
    numeric_medians = metadata["numeric_medians"]

    for col in categorical_columns:
        X_input[col] = X_input[col].fillna("Unknown").astype(str)

    for col in numeric_columns:
        X_input[col] = pd.to_numeric(X_input[col], errors="coerce")
        X_input[col] = X_input[col].fillna(float(numeric_medians.get(col, 0.0)))

    # Ensure numeric NLP components are numeric and preserve exact model order.
    for col in metadata["nlp_feature_names"]:
        X_input[col] = pd.to_numeric(X_input[col], errors="coerce").fillna(0.0)

    return X_input, text_document


# ============================================================
# 3. PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="HR Promotion Prediction",
    page_icon="📈",
    layout="wide",
)

st.title("HR Promotion Prediction")
st.write(
    "Enter employee information and the review text available before the promotion decision. "
    "The app combines structured features with NLP features from the review text."
)

try:
    model, metadata, vectorizer, svd = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

reference_df = load_reference_data()
structured_columns = metadata["structured_columns"]
categorical_columns = metadata["categorical_columns"]
numeric_columns = metadata["numeric_columns"]
numeric_medians = metadata["numeric_medians"]

# Display fields only if the training metadata says they are model features.
with st.form("promotion_prediction_form"):
    st.subheader("Employee information")
    st.caption("Complete the fields using information known before the promotion decision.")

    structured_values = {}
    left, right = st.columns(2)

    for index, col in enumerate(structured_columns):
        container = left if index % 2 == 0 else right
        with container:
            if col in categorical_columns:
                options = []
                if not reference_df.empty and col in reference_df.columns:
                    options = (
                        reference_df[col]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                options = sorted(options)
                if not options:
                    options = ["Unknown"]
                default_value = options[0]
                if col == "Promotion_Eligible" and "Yes" in options:
                    default_value = "Yes"
                selected = st.selectbox(col.replace("_", " "), options, index=options.index(default_value))
                structured_values[col] = selected
            else:
                default_value = float(numeric_medians.get(col, 0.0))
                min_value = 0.0
                max_value = max(default_value * 2.0, 100.0)
                step = 1.0
                if not reference_df.empty and col in reference_df.columns:
                    numeric_series = pd.to_numeric(reference_df[col], errors="coerce").dropna()
                    if not numeric_series.empty:
                        min_value = float(min(0, numeric_series.min()))
                        max_value = float(max(numeric_series.max(), default_value, 1))
                        if max_value == min_value:
                            max_value = min_value + 1.0
                        if any(token in col.lower() for token in ["score", "percentage", "rating"]):
                            step = 0.1
                entered = st.number_input(
                    col.replace("_", " "),
                    min_value=float(min_value),
                    max_value=float(max_value),
                    value=float(np.clip(default_value, min_value, max_value)),
                    step=float(step),
                    key=f"structured_{col}",
                )
                structured_values[col] = entered

    st.subheader("Employee review text (NLP)")
    st.caption("Enter review text from before the promotion decision. Avoid including the final decision or post-decision details.")

    text_values = {}
    text_labels = {
        "Employee_Self_Review_Text": "Employee Self Review",
        "Manager_Performance_Review_Text": "Manager Performance Review",
        "Project_Impact_Summary_Text": "Project Impact Summary",
        "Development_Goals_Text": "Development Goals",
    }
    for col in TEXT_COLS:
        text_values[col] = st.text_area(
            text_labels[col],
            height=110,
            placeholder=f"Enter {text_labels[col].lower()}...",
            key=f"text_{col}",
        )

    submitted = st.form_submit_button("Predict Promotion", type="primary", use_container_width=True)

if submitted:
    if not any(str(text_values.get(col, "")).strip() for col in TEXT_COLS):
        st.error("Enter at least one review text field so NLP can process the employee's text.")
        st.stop()

    try:
        X_input, combined_text = make_prediction_features(
            structured_values,
            text_values,
            metadata,
            vectorizer,
            svd,
        )
        probability = float(model.predict_proba(X_input)[0, 1])
        threshold = float(metadata.get("threshold", 0.5))
        prediction = "Promoted" if probability >= threshold else "Not Promoted"
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.divider()
    st.header("Prediction Results")
    col1, col2 = st.columns(2)
    col1.metric("Model-estimated promotion probability", f"{probability:.1%}")
    col2.metric("Decision threshold", f"{threshold:.2f}")

    if prediction == "Promoted":
        st.success("Model prediction: Promoted")
    else:
        st.info("Model prediction: Not Promoted")

    st.caption(
        "The probability is the model's estimate based on its training data; it is not necessarily "
        "a calibrated real-world probability. The threshold was selected using validation data."
    )

    with st.expander("View submitted information"):
        st.json(structured_values)
        for col in TEXT_COLS:
            st.markdown(f"**{text_labels[col]}**")
            st.write(text_values[col] or "(empty)")

    with st.expander("View NLP feature details"):
        st.write(f"Combined text length: {len(combined_text)} characters")
        st.write(f"TF-IDF((Term Frequency–Inverse Document Frequency) vocabulary size: {len(vectorizer.vocabulary_)}")
