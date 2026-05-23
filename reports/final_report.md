# Final Report: Medical Insurance Cost Prediction

## Executive Summary

The final project pipeline predicts annual medical insurance charges from age,
sex, BMI, number of children, smoking status, and US region. The selected final
model is a tuned `GradientBoostingRegressor`, trained on a deterministic 80/20
split with random seed 42. On the held-out test set, it reaches RMSE 4321.66,
MAE 2439.02, and R2 0.8797.

These results are competitive with published and Kaggle-style work on the same
small medical-cost dataset, but they should not be interpreted as a leaderboard
comparison. Different studies use different preprocessing, split strategies,
outlier treatment, and sometimes transform regression into classification.

## Dataset and Pipeline

The raw dataset comes from Kaggle's Medical Cost Personal Datasets collection:
<https://www.kaggle.com/datasets/mirichoi0218/insurance/data>. It contains
medical insurance charges with demographic and lifestyle predictors. The project
pipeline is:

1. `scripts/ingest_data.py` downloads and verifies the raw Kaggle CSV.
2. `scripts/preprocess_data.py` encodes categorical variables.
3. `scripts/train_model.py` trains a linear regression baseline.
4. `scripts/train_nonlinear_models.py` compares non-linear model families.
5. `scripts/tune_models.py` tunes the strongest model families.
6. `scripts/save_final_model.py` trains and saves the selected final model.
7. `scripts/evaluate_model.py` recreates the test split and exports final
   evaluation artifacts.
8. `scripts/explain_model.py` exports permutation importance and partial
   dependence plots for model interpretation.

## Final Metrics

| Model | Test split | MAE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
| GradientBoostingRegressor | 80/20, seed 42 | 2439.02 | 4321.66 | 0.8797 |

The final model improves over the local baseline linear regression and aligns
with the model-selection stage, where gradient boosting had the strongest RMSE
among the compared non-linear regressors.

## Comparison With Prior Work

| Source | Reported best model/result | Comparison note |
| --- | --- | --- |
| This project | Gradient Boosting: RMSE 4321.66, MAE 2439.02, R2 0.8797 | Reproducible 80/20 split with committed model metadata. |
| Hossen, 2023, ResearchGate report | XGBoost: MAE 2381.567, RMSE 4450.4433, R2 0.8681; Gradient Boosting R2 0.8679 | Similar Kaggle-style dataset, but preprocessing and split details differ. |
| Zhang, Huang, and Wang, 2025, Informatica | Gradient Boosting was reported as best after one-hot encoding, 80/20 split, and tuning; feature ranking emphasized smoking, age, and BMI | Their paper also reports classification-style precision/recall/F1 after transforming the task, so those scores are not directly comparable to this regression table. |

The local model's RMSE is lower than the Hossen XGBoost RMSE while its MAE is
slightly higher. This means the project model reduces larger squared errors on
this split but does not dominate every average-error metric. The R2 values are
close, which is expected for tree boosting methods on this dataset.

## Explainability

The explainability script uses scikit-learn permutation importance, avoiding a
new SHAP dependency. This keeps the environment smaller while still answering
which inputs most affect held-out predictions.

The expected dominant drivers are smoking status, age, and BMI, which is
consistent with Zhang, Huang, and Wang's 2025 feature-importance discussion in
Informatica: <https://doi.org/10.31449/inf.v49i23.8100>. The generated outputs
are:

- `reports/feature_importance.csv`
- `reports/figures/permutation_importance.png`
- `reports/figures/partial_dependence_top_features.png`

## Reproducibility Notes

The project now uses Poetry as the single environment manager. The lock file
captures exact resolved dependencies, and CI installs from `poetry.lock` before
running code quality checks, tests, evaluation, and explainability scripts.

Poetry configuration follows the official Poetry 2.4 documentation for
`pyproject.toml`, non-package mode, and dependency groups:
<https://python-poetry.org/docs/pyproject/> and
<https://python-poetry.org/docs/managing-dependencies/>.
