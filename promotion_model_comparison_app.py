
# ============================================================
# PROMOTION MODEL COMPARISON DASHBOARD
# CatBoost vs Logistic Regression vs Random Forest
# Streamlit + Matplotlib
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Promotion Model Comparison",
    page_icon="📊",
    layout="wide",
)

st.title("Employee Promotion Prediction")
st.subheader("Machine Learning Model Comparison")
st.caption(
    "CatBoost Classifier vs Logistic Regression vs Random Forest Classifier"
)

st.write(
    "Compare classification performance using the test-set metrics "
    "reported by the three training scripts."
)


# ============================================================
# 2. MODEL METRICS
# ============================================================

# Values are taken from the results provided by the user.
comparison = pd.DataFrame({
    "Model": [
        "CatBoost",
        "Logistic Regression",
        "Random Forest",
    ],
    "Accuracy": [
        0.9117,
        0.9050,
        0.8967,
    ],
    "Precision": [
        0.8755,
        0.8427,
        0.8225,
    ],
    "Recall": [
        0.9083,
        0.9375,
        0.9458,
    ],
    "F1-score": [
        0.8916,
        0.8876,
        0.8798,
    ],
    "ROC-AUC": [
        0.9727,
        0.9721,
        0.9665,
    ],
})

# PR-AUC was provided only for CatBoost.
catboost_pr_auc = 0.9565

# Only the CatBoost confusion matrix was provided.
catboost_confusion_matrix = pd.DataFrame(
    [[329, 31], [22, 218]],
    index=["Actual Not Promoted", "Actual Promoted"],
    columns=["Predicted Not Promoted", "Predicted Promoted"],
)


# ============================================================
# 3. SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Information")

st.sidebar.write("**Target:** `Promoted`")
st.sidebar.write("**Classes:** Yes / No")
st.sidebar.write("**Evaluation:** Test-set results")

selected_metrics = st.sidebar.multiselect(
    "Select metrics to display",
    options=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ],
    default=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ],
)

show_table = st.sidebar.checkbox(
    "Show comparison table",
    value=True,
)

show_confusion_matrix = st.sidebar.checkbox(
    "Show CatBoost confusion matrix",
    value=True,
)


# ============================================================
# 4. SUMMARY METRICS
# ============================================================

st.markdown("## Performance Overview")

best_accuracy_model = comparison.loc[
    comparison["Accuracy"].idxmax(), "Model"
]

best_f1_model = comparison.loc[
    comparison["F1-score"].idxmax(), "Model"
]

best_auc_model = comparison.loc[
    comparison["ROC-AUC"].idxmax(), "Model"
]

best_recall_model = comparison.loc[
    comparison["Recall"].idxmax(), "Model"
]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Best Accuracy",
    best_accuracy_model,
    f"{comparison['Accuracy'].max() * 100:.2f}%",
)

col2.metric(
    "Best F1-score",
    best_f1_model,
    f"{comparison['F1-score'].max() * 100:.2f}%",
)

col3.metric(
    "Best ROC-AUC",
    best_auc_model,
    f"{comparison['ROC-AUC'].max():.4f}",
)

col4.metric(
    "Best Recall",
    best_recall_model,
    f"{comparison['Recall'].max() * 100:.2f}%",
)


# ============================================================
# 5. COMPARISON TABLE
# ============================================================

if show_table:
    st.markdown("## Model Metrics Table")

    display_df = comparison.copy()

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ]

    for column in metric_columns:
        display_df[column] = display_df[column].map(
            lambda value: f"{value:.4f}"
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "PR-AUC is not included in the comparison table because "
        "it was supplied only for CatBoost."
    )


# ============================================================
# 6. BAR CHARTS FOR SELECTED METRICS
# ============================================================

st.markdown("## Model Performance Charts")

if selected_metrics:

    chart_data = comparison.set_index("Model")[selected_metrics]

    fig, ax = plt.subplots(figsize=(12, 6))

    chart_data.plot(
        kind="bar",
        ax=ax,
        width=0.78,
    )

    ax.set_title(
        "Classification Metrics Across Models",
        fontsize=15,
    )
    ax.set_xlabel("Machine Learning Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(
        title="Metric",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("Select at least one metric in the sidebar.")


# ============================================================
# 7. INDIVIDUAL METRIC COMPARISON
# ============================================================

st.markdown("## Individual Metric Analysis")

metric_to_explain = st.selectbox(
    "Choose a metric",
    options=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
    ],
)

fig, ax = plt.subplots(figsize=(9, 4.5))

bars = ax.bar(
    comparison["Model"],
    comparison[metric_to_explain],
)

ax.set_title(f"{metric_to_explain} Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax.bar_label(
    bars,
    labels=[
        f"{value:.4f}"
        for value in comparison[metric_to_explain]
    ],
    padding=3,
)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ============================================================
# 8. PR-AUC INFORMATION
# ============================================================

st.markdown("## Precision-Recall AUC")

st.metric(
    "CatBoost PR-AUC",
    f"{catboost_pr_auc:.4f}",
)

st.info(
    "PR-AUC was reported only for CatBoost. The Logistic Regression "
    "and Random Forest PR-AUC values are not available in the supplied "
    "results, so this dashboard does not invent or estimate them."
)


# ============================================================
# 9. CATBOOST CONFUSION MATRIX
# ============================================================

if show_confusion_matrix:

    st.markdown("## CatBoost Confusion Matrix")

    st.write(
        "The confusion matrix below is available only for CatBoost "
        "from the results you supplied."
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))

    matrix_values = catboost_confusion_matrix.values

    image = ax.imshow(matrix_values)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "Not Promoted",
        "Promoted",
    ])

    ax.set_yticklabels([
        "Not Promoted",
        "Promoted",
    ])

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("CatBoost Confusion Matrix")

    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(matrix_values[row, col]),
                ha="center",
                va="center",
                fontsize=14,
            )

    fig.colorbar(image, ax=ax, label="Number of Employees")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        catboost_confusion_matrix,
        use_container_width=True,
    )

    st.write(
        "**Interpretation:** 329 employees were correctly classified "
        "as not promoted, and 218 were correctly classified as promoted. "
        "The model incorrectly predicted promotion for 31 employees and "
        "missed promotion for 22 employees."
    )


# ============================================================
# 10. DOWNLOAD METRICS
# ============================================================

st.markdown("## Export Results")

csv_data = comparison.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Model Comparison CSV",
    data=csv_data,
    file_name="promotion_model_comparison.csv",
    mime="text/csv",
)


# ============================================================
# 11. CONCLUSION
# ============================================================

st.markdown("## Conclusion")

st.write(
    f"**CatBoost** has the highest accuracy "
    f"({comparison.loc[0, 'Accuracy']:.4f}), F1-score "
    f"({comparison.loc[0, 'F1-score']:.4f}) and ROC-AUC "
    f"({comparison.loc[0, 'ROC-AUC']:.4f}) among the supplied results."
)

st.write(
    f"**Random Forest** has the highest recall "
    f"({comparison.loc[2, 'Recall']:.4f}), meaning it identifies "
    "the largest proportion of actual promoted employees at its "
    "current classification threshold."
)

st.write(
    "Logistic Regression offers a competitive result and has higher "
    "recall than CatBoost, but lower precision. The best model depends "
    "on whether your application prioritises overall performance, "
    "identifying more eligible employees, or reducing false positives."
)

st.warning(
    "Ensure all models were evaluated on the exact same test records "
)
