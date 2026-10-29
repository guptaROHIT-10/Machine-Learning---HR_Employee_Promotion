
# # ============================================================
# # HR PROMOTION PREDICTION
# # CatBoost + Hyperparameter Tuning
# # ============================================================

# import json
# import re
# import warnings
# from pathlib import Path

# import numpy as np
# import pandas as pd

# from catboost import CatBoostClassifier

# from sklearn.model_selection import (
#     train_test_split,
#     RandomizedSearchCV,
# )

# from sklearn.metrics import (
#     accuracy_score,
#     average_precision_score,
#     classification_report,
#     confusion_matrix,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
# )

# warnings.filterwarnings("ignore")


# # ============================================================
# # 1. CONFIGURATION
# # ============================================================

# BASE_DIR = Path(__file__).resolve().parent

# DATA_PATH = BASE_DIR / "data" / "promotion_dataset_.csv"

# MODEL_DIR = BASE_DIR / "models"
# OUTPUT_DIR = BASE_DIR / "outputs"

# MODEL_PATH = MODEL_DIR / "promotion_catboost.cbm"
# METADATA_PATH = MODEL_DIR / "promotion_metadata.json"

# METRICS_PATH = OUTPUT_DIR / "promotion_metrics.json"
# PREDICTIONS_PATH = OUTPUT_DIR / "promotion_test_predictions.csv"
# TUNING_PATH = OUTPUT_DIR / "promotion_hyperparameter_results.csv"

# TARGET_COL = "Promoted"

# RANDOM_STATE = 42
# TEST_SIZE = 0.20
# VALIDATION_SIZE = 0.20

# # Add any additional columns that reveal the final promotion outcome.
# # These columns must not be used as prediction features.
# DROP_COLS = [
#     "Employee_ID",
#     "Review_Date",
#     "New_Job_Level_After_Review",
#     "Employee_Self_Review_Text",
#     "Manager_Performance_Review_Text",
#     "Project_Impact_Summary_Text",
#     "Development_Goals_Text",
# ]


# # ============================================================
# # 2. HELPER FUNCTIONS
# # ============================================================

# def convert_target(series):
#     """Convert promotion labels into binary values: 1 or 0."""

#     if pd.api.types.is_numeric_dtype(series):
#         values = set(series.dropna().unique())

#         if values.issubset({0, 1}):
#             return series.astype(int)

#     mapping = {
#         "yes": 1,
#         "promoted": 1,
#         "true": 1,
#         "1": 1,
#         "no": 0,
#         "not promoted": 0,
#         "not_promoted": 0,
#         "false": 0,
#         "0": 0,
#     }

#     converted = (
#         series.astype(str)
#         .str.strip()
#         .str.lower()
#         .map(mapping)
#     )

#     if converted.isna().any():
#         unknown = series[converted.isna()].unique().tolist()

#         raise ValueError(
#             f"Unknown values in {TARGET_COL}: {unknown}. "
#             "Update the target mapping to match your CSV."
#         )

#     return converted.astype(int)


# def calculate_metrics(y_true, probabilities, threshold):
#     """Calculate classification and ranking metrics."""

#     predictions = (probabilities >= threshold).astype(int)

#     results = {
#         "threshold": float(threshold),
#         "accuracy": float(
#             accuracy_score(y_true, predictions)
#         ),
#         "precision_promoted": float(
#             precision_score(
#                 y_true, predictions, zero_division=0
#             )
#         ),
#         "recall_promoted": float(
#             recall_score(
#                 y_true, predictions, zero_division=0
#             )
#         ),
#         "f1_promoted": float(
#             f1_score(
#                 y_true, predictions, zero_division=0
#             )
#         ),
#         "pr_auc": float(
#             average_precision_score(y_true, probabilities)
#         ),
#         "confusion_matrix": confusion_matrix(
#             y_true, predictions, labels=[0, 1]
#         ).tolist(),
#     }

#     if len(np.unique(y_true)) == 2:
#         results["roc_auc"] = float(
#             roc_auc_score(y_true, probabilities)
#         )
#     else:
#         results["roc_auc"] = None

#     return results


# # ============================================================
# # 3. MAIN TRAINING FUNCTION
# # ============================================================

# def main():

#     MODEL_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#     # --------------------------------------------------------
#     # A. Load dataset
#     # --------------------------------------------------------

#     if not DATA_PATH.exists():
#         raise FileNotFoundError(
#             f"Dataset not found: {DATA_PATH}\n"
#             "Put promotion_dataset_.csv inside the data folder."
#         )

#     df = pd.read_csv(DATA_PATH)

#     print("\n========== DATASET INFORMATION ==========")
#     print("Dataset shape:", df.shape)
#     print("Columns:", df.columns.tolist())

#     if TARGET_COL not in df.columns:
#         raise ValueError(
#             f"Target column '{TARGET_COL}' was not found. "
#             f"Available columns: {df.columns.tolist()}"
#         )

#     df = df.dropna(subset=[TARGET_COL]).copy()
#     df = df.reset_index(drop=True)

#     y = convert_target(df[TARGET_COL])

#     if y.nunique() != 2:
#         raise ValueError(
#             "Promoted must contain both positive and negative classes."
#         )

#     if y.value_counts().min() < 5:
#         raise ValueError(
#             "Each class needs at least five examples for this workflow."
#         )

#     print("\nPromotion target distribution:")
#     print(
#         y.value_counts().rename(
#             index={0: "Not Promoted", 1: "Promoted"}
#         )
#     )

#     print("\nTarget percentages:")
#     print(
#         (y.value_counts(normalize=True) * 100)
#         .round(2)
#         .rename(index={0: "Not Promoted", 1: "Promoted"})
#     )

#     # --------------------------------------------------------
#     # B. Prepare feature columns
#     # --------------------------------------------------------

#     excluded_columns = set(DROP_COLS + [TARGET_COL])

#     feature_columns = [
#         col for col in df.columns
#         if col not in excluded_columns
#     ]

#     if not feature_columns:
#         raise ValueError("No usable feature columns were found.")

#     X = df[feature_columns].copy()

#     # Remove completely empty columns.
#     empty_columns = [
#         col for col in X.columns
#         if X[col].isna().all()
#     ]

#     if empty_columns:
#         print("\nDropping entirely empty columns:", empty_columns)

#         X = X.drop(columns=empty_columns)

#         feature_columns = X.columns.tolist()

#     # Automatically identify categorical features.
#     categorical_columns = X.select_dtypes(
#         include=["object", "category", "bool"]
#     ).columns.tolist()

#     numeric_columns = [
#         col for col in X.columns
#         if col not in categorical_columns
#     ]

#     # Normalise categorical features.
#     for col in categorical_columns:
#         X[col] = X[col].fillna("Unknown").astype(str)

#     # Convert numeric features and impute missing values.
#     numeric_medians = {}

#     for col in numeric_columns:
#         X[col] = pd.to_numeric(X[col], errors="coerce")

#         median = X[col].median()

#         if pd.isna(median):
#             median = 0.0

#         numeric_medians[col] = float(median)
#         X[col] = X[col].fillna(median)

#     feature_columns = X.columns.tolist()

#     print("\n========== FEATURES ==========")
#     print("Total features:", len(feature_columns))
#     print("Categorical features:", categorical_columns)
#     print("Numeric features:", numeric_columns)

#     # --------------------------------------------------------
#     # C. Split into train, validation and test sets
#     # --------------------------------------------------------

#     all_indices = np.arange(len(X))

#     train_val_indices, test_indices = train_test_split(
#         all_indices,
#         test_size=TEST_SIZE,
#         random_state=RANDOM_STATE,
#         stratify=y,
#     )

#     y_train_val = y.iloc[train_val_indices]

#     relative_validation_size = (
#         VALIDATION_SIZE / (1.0 - TEST_SIZE)
#     )

#     train_indices, validation_indices = train_test_split(
#         train_val_indices,
#         test_size=relative_validation_size,
#         random_state=RANDOM_STATE,
#         stratify=y_train_val,
#     )

#     X_train = X.iloc[train_indices].reset_index(drop=True)
#     X_validation = X.iloc[validation_indices].reset_index(drop=True)
#     X_test = X.iloc[test_indices].reset_index(drop=True)

#     y_train = y.iloc[train_indices].reset_index(drop=True)
#     y_validation = y.iloc[validation_indices].reset_index(drop=True)
#     y_test = y.iloc[test_indices].reset_index(drop=True)

#     print("\n========== DATA SPLITS ==========")
#     print("Training rows:", len(X_train))
#     print("Validation rows:", len(X_validation))
#     print("Test rows:", len(X_test))

#     # --------------------------------------------------------
#     # D. Hyperparameter tuning
#     # --------------------------------------------------------

#     print("\n========== HYPERPARAMETER TUNING ==========")

#     base_model = CatBoostClassifier(
#         loss_function="Logloss",
#         auto_class_weights="Balanced",
#         random_seed=RANDOM_STATE,
#         verbose=False,
#         allow_writing_files=False,
#         thread_count=1,
#     )

#     param_distributions = {
#         "iterations": [300, 500, 700, 900],
#         "depth": [4, 5, 6, 7, 8],
#         "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.10],
#         "l2_leaf_reg": [3, 5, 7, 9],
#         "random_strength": [0.5, 1, 2],
#     }

#     search = RandomizedSearchCV(
#         estimator=base_model,
#         param_distributions=param_distributions,
#         n_iter=12,
#         scoring="average_precision",
#         cv=3,
#         random_state=RANDOM_STATE,
#         n_jobs=1,
#         verbose=2,
#         refit=True,
#         error_score="raise",
#     )

#     search.fit(
#         X_train,
#         y_train,
#         cat_features=categorical_columns,
#     )

#     print("\nBest hyperparameters:")
#     print(search.best_params_)

#     print(
#         "Best cross-validation average precision:",
#         round(search.best_score_, 4),
#     )

#     tuning_results = pd.DataFrame(search.cv_results_)
#     tuning_results = tuning_results.sort_values(
#         "rank_test_score"
#     )

#     tuning_results.to_csv(TUNING_PATH, index=False)

#     # --------------------------------------------------------
#     # E. Train final model using the best parameters
#     # --------------------------------------------------------

#     print("\n========== FINAL MODEL TRAINING ==========")

#     model = CatBoostClassifier(
#         **search.best_params_,
#         loss_function="Logloss",
#         eval_metric="AUC",
#         auto_class_weights="Balanced",
#         random_seed=RANDOM_STATE,
#         verbose=100,
#         allow_writing_files=False,
#     )

#     model.fit(
#         X_train,
#         y_train,
#         cat_features=categorical_columns,
#         eval_set=(X_validation, y_validation),
#         early_stopping_rounds=100,
#         verbose=100,
#     )

#     print("Best iteration:", model.get_best_iteration())

#     # --------------------------------------------------------
#     # F. Select threshold using validation data only
#     # --------------------------------------------------------

#     print("\n========== THRESHOLD SELECTION ==========")

#     validation_probabilities = model.predict_proba(
#         X_validation
#     )[:, 1]

#     threshold_candidates = np.arange(0.10, 0.91, 0.01)

#     threshold_results = []

#     for threshold in threshold_candidates:

#         predictions = (
#             validation_probabilities >= threshold
#         ).astype(int)

#         threshold_results.append({
#             "threshold": float(threshold),
#             "f1": float(
#                 f1_score(
#                     y_validation,
#                     predictions,
#                     zero_division=0,
#                 )
#             ),
#             "precision": float(
#                 precision_score(
#                     y_validation,
#                     predictions,
#                     zero_division=0,
#                 )
#             ),
#             "recall": float(
#                 recall_score(
#                     y_validation,
#                     predictions,
#                     zero_division=0,
#                 )
#             ),
#         })

#     best_result = max(
#         threshold_results,
#         key=lambda result: result["f1"],
#     )

#     selected_threshold = best_result["threshold"]

#     print("Selected threshold:", round(selected_threshold, 2))
#     print("Validation F1:", round(best_result["f1"], 4))
#     print("Validation precision:", round(best_result["precision"], 4))
#     print("Validation recall:", round(best_result["recall"], 4))

#     # --------------------------------------------------------
#     # G. Evaluate on the held-out test set
#     # --------------------------------------------------------

#     print("\n========== TEST RESULTS ==========")

#     test_probabilities = model.predict_proba(X_test)[:, 1]

#     test_predictions = (
#         test_probabilities >= selected_threshold
#     ).astype(int)

#     metrics = calculate_metrics(
#         y_test,
#         test_probabilities,
#         selected_threshold,
#     )

#     print(f"Accuracy:  {metrics['accuracy']:.4f}")
#     print(f"Precision: {metrics['precision_promoted']:.4f}")
#     print(f"Recall:    {metrics['recall_promoted']:.4f}")
#     print(f"F1 score:  {metrics['f1_promoted']:.4f}")
#     print(f"PR-AUC:    {metrics['pr_auc']:.4f}")

#     if metrics["roc_auc"] is not None:
#         print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

#     print("\nConfusion matrix:")
#     print(
#         pd.DataFrame(
#             metrics["confusion_matrix"],
#             index=["Actual Not Promoted", "Actual Promoted"],
#             columns=["Predicted Not Promoted", "Predicted Promoted"],
#         )
#     )

#     print("\nClassification report:")
#     print(
#         classification_report(
#             y_test,
#             test_predictions,
#             labels=[0, 1],
#             target_names=["Not Promoted", "Promoted"],
#             zero_division=0,
#         )
#     )

#     # --------------------------------------------------------
#     # H. Save model and metadata
#     # --------------------------------------------------------

#     model.save_model(str(MODEL_PATH))

#     metadata = {
#         "target_column": TARGET_COL,
#         "feature_columns": feature_columns,
#         "categorical_columns": categorical_columns,
#         "numeric_columns": numeric_columns,
#         "numeric_medians": numeric_medians,
#         "threshold": float(selected_threshold),
#         "random_state": RANDOM_STATE,
#         "test_size": TEST_SIZE,
#         "validation_size": VALIDATION_SIZE,
#         "best_params": search.best_params_,
#         "model_type": "CatBoostClassifier",
#         "synthetic_data_warning": (
#             "If trained on synthetic data, test metrics do not establish "
#             "real-world promotion prediction performance."
#         ),
#     }

#     with open(METADATA_PATH, "w", encoding="utf-8") as file:
#         json.dump(metadata, file, indent=4)

#     with open(METRICS_PATH, "w", encoding="utf-8") as file:
#         json.dump(
#             {
#                 "validation": {
#                     "selected_threshold": float(selected_threshold),
#                     "f1": float(best_result["f1"]),
#                     "precision": float(best_result["precision"]),
#                     "recall": float(best_result["recall"]),
#                 },
#                 "test": metrics,
#                 "best_params": search.best_params_,
#                 "best_cv_average_precision": float(search.best_score_),
#             },
#             file,
#             indent=4,
#         )

#     # --------------------------------------------------------
#     # I. Save test predictions
#     # --------------------------------------------------------

#     output_df = df.iloc[test_indices].copy().reset_index(drop=True)

#     output_df["Actual_Promoted"] = y_test.map({
#         0: "No",
#         1: "Yes",
#     }).values

#     output_df["Predicted_Promoted"] = pd.Series(
#         test_predictions
#     ).map({
#         0: "No",
#         1: "Yes",
#     }).values

#     output_df["Promotion_Probability"] = test_probabilities

#     output_df.to_csv(PREDICTIONS_PATH, index=False)

#     # --------------------------------------------------------
#     # J. Summary
#     # --------------------------------------------------------

#     print("\n========== FILES SAVED ==========")
#     print("Model:", MODEL_PATH)
#     print("Metadata:", METADATA_PATH)
#     print("Metrics:", METRICS_PATH)
#     print("Test predictions:", PREDICTIONS_PATH)
#     print("Tuning results:", TUNING_PATH)

#     print("\nPromotion model training completed.")


# if __name__ == "__main__":
#     main()


# ============================================================
# HR PROMOTION PREDICTION: STRUCTURED DATA + NLP
# CatBoost + TF-IDF + TruncatedSVD + Hyperparameter Tuning
# ============================================================

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "promotion_dataset_.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "promotion_catboost_nlp.cbm"
METADATA_PATH = MODEL_DIR / "promotion_nlp_metadata.json"
VECTORIZER_PATH = MODEL_DIR / "promotion_tfidf_vectorizer.joblib"
SVD_PATH = MODEL_DIR / "promotion_svd.joblib"
METRICS_PATH = OUTPUT_DIR / "promotion_nlp_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "promotion_nlp_test_predictions.csv"
TUNING_PATH = OUTPUT_DIR / "promotion_nlp_hyperparameter_results.csv"

TARGET_COL = "Promoted"
TEXT_COLS = [
    "Employee_Self_Review_Text",
    "Manager_Performance_Review_Text",
    "Project_Impact_Summary_Text",
    "Development_Goals_Text",
]

# Exclude post-decision fields and identifiers. Review_Date is excluded too.
DROP_COLS = [
    "Employee_ID",
    "Review_Date",
    "New_Job_Level_After_Review",
    "Promotion_Recommendation",
    "Promotion_Status",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
SVD_COMPONENTS = 30


# ============================================================
# 2. HELPERS
# ============================================================
def convert_target(series):
    """Convert promotion labels to binary values: Promoted=1, No=0."""
    if pd.api.types.is_numeric_dtype(series):
        values = set(series.dropna().unique())
        if values.issubset({0, 1}):
            return series.astype(int)

    mapping = {
        "yes": 1,
        "promoted": 1,
        "true": 1,
        "1": 1,
        "no": 0,
        "not promoted": 0,
        "not_promoted": 0,
        "false": 0,
        "0": 0,
    }
    converted = series.astype(str).str.strip().str.lower().map(mapping)
    if converted.isna().any():
        unknown = series[converted.isna()].unique().tolist()
        raise ValueError(f"Unknown values in {TARGET_COL}: {unknown}")
    return converted.astype(int)


def combine_text_rows(frame):
    """Combine available text fields for each employee into one NLP document."""
    missing = [col for col in TEXT_COLS if col not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing NLP columns: {missing}. Check the CSV column names."
        )
    clean_frame = frame[TEXT_COLS].fillna("").astype(str)
    return clean_frame.agg(" ".join, axis=1).tolist()


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_promoted": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall_promoted": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_promoted": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, probabilities))
        if len(np.unique(y_true)) == 2 else None
    )
    return metrics


def transform_text(vectorizer, svd, texts):
    tfidf = vectorizer.transform(texts)
    return svd.transform(tfidf)


# ============================================================
# 3. MAIN
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
    print("\n========== DATASET INFORMATION ==========")
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    required = [TARGET_COL] + TEXT_COLS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    y = convert_target(df[TARGET_COL]).reset_index(drop=True)
    if y.nunique() != 2:
        raise ValueError("Promoted must contain both Yes and No classes.")
    if y.value_counts().min() < 5:
        raise ValueError("Each target class needs at least five rows.")

    print("\nTarget distribution:")
    print(y.value_counts().rename(index={0: "Not Promoted", 1: "Promoted"}))

    # Split before fitting TF-IDF or SVD to prevent text vocabulary leakage.
    all_indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        all_indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    y_train_val = y.iloc[train_val_idx]
    relative_val_size = VALIDATION_SIZE / (1.0 - TEST_SIZE)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    print("\n========== DATA SPLITS ==========")
    print(f"Training: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")

    # --------------------------------------------------------
    # A. NLP: TF-IDF fitted on training text only, then SVD
    # --------------------------------------------------------
    train_texts = combine_text_rows(train_df)
    val_texts = combine_text_rows(val_df)
    test_texts = combine_text_rows(test_df)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        sublinear_tf=True,
    )
    try:
        train_tfidf = vectorizer.fit_transform(train_texts)
    except ValueError as exc:
        raise ValueError(
            "TF-IDF could not create a vocabulary. Check that the text columns "
            "contain meaningful text."
        ) from exc

    n_components = min(SVD_COMPONENTS, train_tfidf.shape[1] - 1, train_tfidf.shape[0] - 1)
    if n_components < 2:
        raise ValueError("Not enough text vocabulary to create SVD NLP features.")

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    train_text_features = svd.fit_transform(train_tfidf)
    val_text_features = transform_text(vectorizer, svd, val_texts)
    test_text_features = transform_text(vectorizer, svd, test_texts)

    nlp_feature_names = [f"NLP_SVD_{i + 1}" for i in range(n_components)]
    print("\n========== NLP FEATURES ==========")
    print("Text columns combined:", TEXT_COLS)
    print("TF-IDF vocabulary size:", len(vectorizer.vocabulary_))
    print("SVD components created:", n_components)

    # --------------------------------------------------------
    # B. Structured features (exclude target, text, IDs, post-decision fields)
    # --------------------------------------------------------
    excluded = set(DROP_COLS + TEXT_COLS + [TARGET_COL])
    structured_columns = [col for col in df.columns if col not in excluded]
    if not structured_columns:
        raise ValueError("No structured feature columns remain after exclusions.")

    X_train = train_df[structured_columns].copy()
    X_val = val_df[structured_columns].copy()
    X_test = test_df[structured_columns].copy()

    # Identify categorical fields from the training partition.
    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    for col in categorical_columns:
        X_train[col] = X_train[col].fillna("Unknown").astype(str)
        X_val[col] = X_val[col].fillna("Unknown").astype(str)
        X_test[col] = X_test[col].fillna("Unknown").astype(str)

    numeric_columns = [col for col in X_train.columns if col not in categorical_columns]
    numeric_medians = {}
    for col in numeric_columns:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_val[col] = pd.to_numeric(X_val[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        median = X_train[col].median()
        if pd.isna(median):
            median = 0.0
        numeric_medians[col] = float(median)
        X_train[col] = X_train[col].fillna(median)
        X_val[col] = X_val[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    # Add dense NLP components to structured data.
    for i, name in enumerate(nlp_feature_names):
        X_train[name] = train_text_features[:, i]
        X_val[name] = val_text_features[:, i]
        X_test[name] = test_text_features[:, i]

    feature_columns = X_train.columns.tolist()
    print("\nStructured features:", structured_columns)
    print("Total model features including NLP:", len(feature_columns))

    # --------------------------------------------------------
    # C. Hyperparameter tuning on training split only
    # --------------------------------------------------------
    print("\n========== HYPERPARAMETER TUNING ==========")
    base_model = CatBoostClassifier(
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )
    param_distributions = {
        "iterations": [300, 500, 700, 900],
        "depth": [4, 5, 6, 7, 8],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.10],
        "l2_leaf_reg": [3, 5, 7, 9],
        "random_strength": [0.5, 1, 2],
    }
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=12,
        scoring="average_precision",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=2,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train, cat_features=categorical_columns)
    print("Best parameters:", search.best_params_)
    print("Best CV average precision:", round(search.best_score_, 4))
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
        TUNING_PATH, index=False
    )

    # Fit chosen configuration using validation for early stopping.
    model = CatBoostClassifier(
        **search.best_params_,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=100,
        allow_writing_files=False,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=categorical_columns,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=100,
    )

    # --------------------------------------------------------
    # D. Select threshold on validation only
    # --------------------------------------------------------
    val_probabilities = model.predict_proba(X_val)[:, 1]
    threshold_results = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        val_pred = (val_probabilities >= threshold).astype(int)
        threshold_results.append({
            "threshold": float(threshold),
            "f1": float(f1_score(y_val, val_pred, zero_division=0)),
            "precision": float(precision_score(y_val, val_pred, zero_division=0)),
            "recall": float(recall_score(y_val, val_pred, zero_division=0)),
        })
    best_threshold_result = max(threshold_results, key=lambda item: item["f1"])
    threshold = best_threshold_result["threshold"]
    print("\nSelected validation threshold:", round(threshold, 2))
    print("Validation F1:", round(best_threshold_result["f1"], 4))
    print("Validation precision:", round(best_threshold_result["precision"], 4))
    print("Validation recall:", round(best_threshold_result["recall"], 4))

    # --------------------------------------------------------
    # E. Final test evaluation
    # --------------------------------------------------------
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)
    metrics = calculate_metrics(y_test, test_probabilities, threshold)

    print("\n========== TEST RESULTS ==========")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_promoted']:.4f}")
    print(f"Recall:    {metrics['recall_promoted']:.4f}")
    print(f"F1 score:  {metrics['f1_promoted']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nConfusion matrix:")
    print(pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual Not Promoted", "Actual Promoted"],
        columns=["Predicted Not Promoted", "Predicted Promoted"],
    ))
    print("\nClassification report:")
    print(classification_report(
        y_test, test_predictions,
        labels=[0, 1],
        target_names=["Not Promoted", "Promoted"],
        zero_division=0,
    ))

    # --------------------------------------------------------
    # F. Save artifacts
    # --------------------------------------------------------
    model.save_model(str(MODEL_PATH))
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(svd, SVD_PATH)

    metadata = {
        "target_column": TARGET_COL,
        "feature_columns": feature_columns,
        "structured_columns": structured_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "numeric_medians": numeric_medians,
        "text_columns": TEXT_COLS,
        "nlp_feature_names": nlp_feature_names,
        "tfidf_max_features": 10000,
        "svd_components": n_components,
        "threshold": float(threshold),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "best_params": search.best_params_,
        "model_type": "CatBoost with TF-IDF + TruncatedSVD NLP features",
        "warning": (
            "Synthetic-data metrics do not establish real-world promotion "
            "prediction performance. Only use text available before the decision."
        ),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump({
            "validation": {
                "selected_threshold": float(threshold),
                "f1": float(best_threshold_result["f1"]),
                "precision": float(best_threshold_result["precision"]),
                "recall": float(best_threshold_result["recall"]),
            },
            "test": metrics,
            "best_params": search.best_params_,
            "best_cv_average_precision": float(search.best_score_),
        }, file, indent=4)

    output_df = test_df.copy()
    output_df["Actual_Promoted"] = y_test.map({0: "No", 1: "Yes"}).values
    output_df["Predicted_Promoted"] = pd.Series(test_predictions).map(
        {0: "No", 1: "Yes"}
    ).values
    output_df["Promotion_Probability"] = test_probabilities
    output_df.to_csv(PREDICTIONS_PATH, index=False)

    print("\n========== FILES SAVED ==========")
    print("Model:", MODEL_PATH)
    print("Metadata:", METADATA_PATH)
    print("TF-IDF vectorizer:", VECTORIZER_PATH)
    print("SVD transformer:", SVD_PATH)
    print("Metrics:", METRICS_PATH)
    print("Test predictions:", PREDICTIONS_PATH)
    print("Tuning results:", TUNING_PATH)
    print("\nPromotion NLP model training completed.")


if __name__ == "__main__":
    main()
