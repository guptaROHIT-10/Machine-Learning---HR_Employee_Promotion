# Employee Promotion Prediction System

A machine learning project that predicts employee promotion outcomes using employee performance, experience, eligibility, skills, leadership assessments, and review text.

The project compares classification algorithms to understand how effectively they identify employees who are promoted.

## Features

- Predicts employee promotion outcomes.
- Analyses performance ratings, KPI scores, training, experience, and leadership scores.
- Processes employee self-reviews and manager feedback using NLP.
- Uses TF-IDF and Truncated SVD to extract text features.
- Supports CatBoost, Logistic Regression, and Random Forest model evaluation.
- Reports Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC.
- Provides confusion matrices and model comparison visualisations.
- Uses class-aware evaluation for the imbalanced target.

## Tech Stack

- **Language:** Python
- **Machine Learning:** scikit-learn, CatBoost
- **NLP:** TF-IDF, Truncated SVD
- **Data Processing:** Pandas, NumPy
- **Visualisation:** Matplotlib
- **Web Application:** Streamlit
- **Model Persistence:** Joblib

## Dataset

The promotion dataset contains employee-related features such as:

- Department and job role
- Job level and years at the company
- Years in the current role
- Performance rating and KPI score
- Training hours and awards
- Leadership and skills assessment scores
- Attendance percentage
- Employee self-review
- Manager performance review
- Project impact summary
- Development goals
- Promotion eligibility

**Target variable:** `Promoted`

The target represents whether an employee was promoted.

Any fields created after the promotion decision must be excluded from training to prevent target leakage.

## Project Structure

```text
HR-Promotion/
├── data/
│   └── promotion_dataset_.csv
├── models/
├── outputs/
├── promotion_app.py
├── train_promotion_logistic.py
├── train_promotion_random_forest.py
└── README.md
```

*Update this structure to reflect your actual files and trained models.*

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone <your-repository-url>
cd HR-Promotion
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
python -m pip install pandas numpy scikit-learn catboost streamlit matplotlib joblib
```

## Run the Application

```bash
python -m streamlit run promotion_app.py
```

## Train the Models

Train the Logistic Regression model:

```bash
python train_promotion_logistic.py
```

Train the Random Forest model:

```bash
python train_promotion_random_forest.py
```

Run the corresponding CatBoost training script if it is included in your repository.

## Model Evaluation

Previously reported test results:

| Metric | CatBoost | Logistic Regression | Random Forest |
|---|---:|---:|---:|
| Accuracy | 91.17% | 90.50% | 89.67% |
| Precision | 87.55% | 84.27% | 82.25% |
| Recall | 90.83% | 93.75% | 94.58% |
| F1-score | 89.16% | 88.76% | 87.98% |
| ROC-AUC | 0.9727 | 0.9721 | 0.9665 |
| PR-AUC | 0.9565 | Not provided | Not provided |

*These results were reported from earlier evaluations. Confirm that all models use the same test records, target definition, and leakage-free features before making a direct comparison.*

### Observations

- CatBoost has the highest reported accuracy and F1-score.
- Logistic Regression has the highest reported precision among the three models only if compared on the supplied figures? No: CatBoost has the higher precision.
- Random Forest has the highest reported recall.
- CatBoost has the highest reported ROC-AUC.

## Future Improvements

- Validate the dataset against real employee outcomes.
- Improve model calibration and threshold selection.
- Add feature importance and explainability tools.
- Evaluate fairness across relevant employee groups.
- Test performance on data from a different review period.

## Disclaimer

This project is intended for educational and experimental purposes. Promotion decisions involve human judgement, organisational policies, and individual circumstances. Model predictions should support, not replace, a fair and transparent human review process.
