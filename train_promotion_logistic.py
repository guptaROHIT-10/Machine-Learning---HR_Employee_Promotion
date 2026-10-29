
# ============================================================
# EMPLOYEE PROMOTION PREDICTION
# Logistic Regression + NLP + Hyperparameter Tuning
# Dataset: data/promotion_dataset_.csv
# ============================================================

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "promotion_dataset_.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "promotion_logistic_nlp.joblib"
VECTORIZER_PATH = MODEL_DIR / "promotion_logistic_tfidf.joblib"
SVD_PATH = MODEL_DIR / "promotion_logistic_svd.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "promotion_logistic_preprocessor.joblib"
METADATA_PATH = MODEL_DIR / "promotion_logistic_metadata.json"

METRICS_PATH = OUTPUT_DIR / "promotion_logistic_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "promotion_logistic_predictions.csv"
TUNING_PATH = OUTPUT_DIR / "promotion_logistic_tuning_results.csv"

TARGET_COL = "Promoted"

TEXT_COLS = [
    "Employee_Self_Review_Text",
    "Manager_Performance_Review_Text",
    "Project_Impact_Summary_Text",
    "Development_Goals_Text",
]

# Exclude IDs, dates and post-decision outcomes.
DROP_COLS = [
    "Employee_ID",
    "Review_Date",
    "New_Job_Level_After_Review",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
N_SVD_COMPONENTS = 30


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).lower().split())


def make_text_corpus(frame):
    return [
        " ".join(clean_text(row[col]) for col in TEXT_COLS)
        for _, row in frame.iterrows()
    ]


def calculate_metrics(y_true, predictions, probabilities):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


# ============================================================
# 3. LOAD DATASET
# ============================================================

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place promotion_dataset_.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [TARGET_COL] + TEXT_COLS
    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TARGET_COL]).copy()

    # Normalise target labels.
    df[TARGET_COL] = (
        df[TARGET_COL].astype(str).str.strip().str.lower()
    )

    label_mapping = {
        "no": 0,
        "yes": 1,
    }

    df[TARGET_COL] = df[TARGET_COL].map(label_mapping)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    for col in TEXT_COLS:
        df[col] = df[col].fillna("").astype(str)

    if df[TARGET_COL].nunique() != 2:
        raise ValueError(
            "Promoted must contain both Yes and No labels."
        )

    print("\n========== DATASET ==========")
    print("Dataset shape:", df.shape)
    print("\nTarget distribution:")
    print(df[TARGET_COL].value_counts().rename(
        index={0: "No", 1: "Yes"}
    ))

    y = df[TARGET_COL]


    # ========================================================
    # 4. SPLIT DATA: 60% TRAIN, 20% VALIDATION, 20% TEST
    # ========================================================

    indices = np.arange(len(df))

    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    relative_validation_size = VALIDATION_SIZE / (1 - TEST_SIZE)

    train_indices, validation_indices = train_test_split(
        train_val_indices,
        test_size=relative_validation_size,
        random_state=RANDOM_STATE,
        stratify=y.iloc[train_val_indices],
    )

    train_df = df.iloc[train_indices].copy().reset_index(drop=True)
    validation_df = (
        df.iloc[validation_indices].copy().reset_index(drop=True)
    )
    test_df = df.iloc[test_indices].copy().reset_index(drop=True)

    y_train = train_df[TARGET_COL]
    y_validation = validation_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    print("\n========== DATA SPLITS ==========")
    print("Training rows:", len(train_df))
    print("Validation rows:", len(validation_df))
    print("Testing rows:", len(test_df))


    # ========================================================
    # 5. SELECT STRUCTURED FEATURES
    # ========================================================

    excluded_columns = set(
        DROP_COLS + TEXT_COLS + [TARGET_COL]
    )

    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]

    X_train = train_df[feature_columns].copy()
    X_validation = validation_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_columns = [
        col for col in feature_columns
        if col not in categorical_columns
    ]

    for col in numeric_columns:
        X_train[col] = pd.to_numeric(
            X_train[col], errors="coerce"
        )
        X_validation[col] = pd.to_numeric(
            X_validation[col], errors="coerce"
        )
        X_test[col] = pd.to_numeric(
            X_test[col], errors="coerce"
        )

    print("\nStructured features:", len(feature_columns))
    print("Categorical columns:", categorical_columns)
    print("Numeric columns:", numeric_columns)


    # ========================================================
    # 6. NLP: TF-IDF + TRUNCATED SVD
    # Fit on training text only.
    # ========================================================

    train_corpus = make_text_corpus(train_df)
    validation_corpus = make_text_corpus(validation_df)
    test_corpus = make_text_corpus(test_df)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        min_df=2,
    )

    try:
        train_tfidf = vectorizer.fit_transform(train_corpus)
        validation_tfidf = vectorizer.transform(validation_corpus)
        test_tfidf = vectorizer.transform(test_corpus)
    except ValueError as exc:
        raise ValueError(
            "TF-IDF could not build a vocabulary. "
            "Check the promotion review text columns."
        ) from exc

    n_components = min(
        N_SVD_COMPONENTS,
        train_tfidf.shape[1] - 1,
        train_tfidf.shape[0] - 1,
    )

    if n_components < 1:
        raise ValueError("Insufficient text data for TruncatedSVD.")

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=RANDOM_STATE,
    )

    train_text_features = svd.fit_transform(train_tfidf)
    validation_text_features = svd.transform(validation_tfidf)
    test_text_features = svd.transform(test_tfidf)

    svd_columns = [
        f"NLP_SVD_{i + 1}" for i in range(n_components)
    ]

    print("\nTF-IDF vocabulary size:", len(vectorizer.vocabulary_))
    print("SVD NLP features:", n_components)
    print(
        "Explained variance ratio:",
        round(float(svd.explained_variance_ratio_.sum()), 4),
    )


    # ========================================================
    # 7. PREPROCESS STRUCTURED FEATURES
    # ========================================================

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    train_structured = preprocessor.fit_transform(X_train)
    validation_structured = preprocessor.transform(X_validation)
    test_structured = preprocessor.transform(X_test)

    # Combine structured features with NLP features.
    train_combined = hstack(
        [
            csr_matrix(train_structured),
            csr_matrix(train_text_features),
        ],
        format="csr",
    )

    validation_combined = hstack(
        [
            csr_matrix(validation_structured),
            csr_matrix(validation_text_features),
        ],
        format="csr",
    )

    test_combined = hstack(
        [
            csr_matrix(test_structured),
            csr_matrix(test_text_features),
        ],
        format="csr",
    )

    print("\nTraining matrix:", train_combined.shape)
    print("Validation matrix:", validation_combined.shape)
    print("Testing matrix:", test_combined.shape)


    # ========================================================
    # 8. LOGISTIC REGRESSION HYPERPARAMETER TUNING
    # ========================================================

    print("\n========== HYPERPARAMETER TUNING ==========")

    base_model = LogisticRegression(
        max_iter=3000,
        random_state=RANDOM_STATE,
    )

    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "class_weight": [None, "balanced"],
        "solver": ["lbfgs"],
    }

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=2,
        refit=True,
        error_score="raise",
    )

    search.fit(train_combined, y_train)

    print("\nBest hyperparameters:")
    print(search.best_params_)
    print("Best cross-validation F1:", round(search.best_score_, 4))

    tuning_results = pd.DataFrame(search.cv_results_)
    tuning_results = tuning_results.sort_values("rank_test_score")
    tuning_results.to_csv(TUNING_PATH, index=False)


    # ========================================================
    # 9. TRAIN FINAL MODEL
    # ========================================================

    print("\n========== TRAINING FINAL MODEL ==========")

    model = LogisticRegression(
        **search.best_params_,
        max_iter=3000,
        random_state=RANDOM_STATE,
    )

    model.fit(train_combined, y_train)

    print("Training completed.")


    # ========================================================
    # 10. VALIDATION AND TEST METRICS
    # ========================================================

    validation_predictions = model.predict(validation_combined)
    validation_probabilities = model.predict_proba(
        validation_combined
    )[:, 1]

    test_predictions = model.predict(test_combined)
    test_probabilities = model.predict_proba(test_combined)[:, 1]

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
        validation_probabilities,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        test_probabilities,
    )

    print("\n========== VALIDATION RESULTS ==========")
    for name, value in validation_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n========== TEST RESULTS ==========")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-score:  {test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")

    print("\n========== CLASSIFICATION REPORT ==========")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=[0, 1],
            target_names=["Not Promoted", "Promoted"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    )

    print("Confusion Matrix:")
    print(matrix)


    # ========================================================
    # 11. SAVE MODEL AND PREPROCESSING OBJECTS
    # ========================================================

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(svd, SVD_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    metadata = {
        "model_type": "LogisticRegression",
        "target_column": TARGET_COL,
        "label_mapping": {"No": 0, "Yes": 1},
        "positive_class": 1,
        "text_columns": TEXT_COLS,
        "dropped_columns": DROP_COLS,
        "structured_feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "svd_columns": svd_columns,
        "svd_components": n_components,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "best_params": search.best_params_,
        "warning": (
            "Synthetic-data metrics do not establish real-world "
            "promotion prediction performance."
        ),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "confusion_matrix": matrix.tolist(),
                "best_cv_f1": float(search.best_score_),
                "best_params": search.best_params_,
            },
            file,
            indent=4,
        )


    # ========================================================
    # 12. SAVE TEST PREDICTIONS
    # ========================================================

    output_df = test_df.copy()

    output_df["Actual_Promoted"] = y_test.map(
        {0: "No", 1: "Yes"}
    ).to_numpy()

    output_df["Predicted_Promoted"] = pd.Series(
        test_predictions
    ).map({0: "No", 1: "Yes"}).to_numpy()

    output_df["Promotion_Probability"] = test_probabilities

    output_df.to_csv(PREDICTIONS_PATH, index=False)


    # ========================================================
    # 13. COMPLETION SUMMARY
    # ========================================================

    print("\n========== FILES SAVED ==========")
    print("Model:", MODEL_PATH)
    print("Vectorizer:", VECTORIZER_PATH)
    print("SVD:", SVD_PATH)
    print("Preprocessor:", PREPROCESSOR_PATH)
    print("Metadata:", METADATA_PATH)
    print("Metrics:", METRICS_PATH)
    print("Predictions:", PREDICTIONS_PATH)
    print("Tuning results:", TUNING_PATH)

    print("\nLogistic Regression training completed successfully.")


if __name__ == "__main__":
    main()
