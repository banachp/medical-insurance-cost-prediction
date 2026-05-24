# Medical Insurance Cost Prediction

This repository is a final assignment for the Reproducible Research course at
the University of Warsaw. It builds a reproducible machine learning pipeline for
predicting annual medical insurance charges from demographic and lifestyle
features.

## Project Goal

Predict medical insurance costs using the
[Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)
dataset from Kaggle, then evaluate and explain the final model in a way that can
be rerun by every contributor.

## Environment

The project uses Poetry as the only supported environment manager. Install
Poetry, then create the locked environment:

```bash
poetry install
```

Poetry configuration is stored in `pyproject.toml`, and exact resolved package
versions are stored in `poetry.lock`.

## Downloading the Data

The repository already contains the dataset used for the submitted results. To
download it again from Kaggle:

1. In Kaggle, go to **Settings -> API -> Generate New Token**.
2. Create `.kaggle/kaggle.json` at the root of this repo:

   ```json
   {"username": "your_kaggle_username", "key": "your_kaggle_api_key"}
   ```

3. Run:

   ```bash
   poetry run python scripts/ingest_data.py
   ```

The raw dataset will be saved to `data/insurance.csv`.

## Reproducible Pipeline

Run the pipeline from the repository root:

```bash
poetry run python scripts/preprocess_data.py
poetry run python scripts/train_model.py
poetry run python scripts/train_nonlinear_models.py
poetry run python scripts/tune_models.py
poetry run python scripts/save_final_model.py
poetry run python scripts/evaluate_model.py
poetry run python scripts/explain_model.py
```

The final model is saved at `models/final_model.joblib`. Final evaluation and
explainability artifacts are saved under `reports/`.

## Final Results

The selected final model is a tuned `GradientBoostingRegressor`, evaluated on a
deterministic 80/20 split with random seed 42.

| Metric | Value |
| --- | ---: |
| MAE | 2439.02 |
| RMSE | 4321.66 |
| R2 | 0.8797 |

See `reports/final_report.md` for the comparison with related Kaggle-style work
and published results.

## Exploratory Data Analysis

The notebook `notebooks/eda.ipynb` provides a walkthrough of the dataset before
modelling. It covers:

- data quality checks
- target distribution
- numeric and categorical feature distributions
- charges by demographic and lifestyle groups
- correlation analysis

Figures are saved to `reports/figures/`, and the exported HTML report is saved
to `reports/eda_report.html`.

## Validation

Before opening a pull request, run:

```bash
poetry check --lock --strict
poetry install
poetry run black --check .
poetry run isort --check-only .
poetry run flake8 .
poetry run nbstripout --verify notebooks/eda.ipynb
poetry run pytest
poetry run python scripts/evaluate_model.py
poetry run python scripts/explain_model.py
```

## Branch and PR Rules

When creating a new branch, start with your name and specify the task. Example:
`igor/reproducible-evaluation-reporting`.

Repository collaboration rules:

1. Do not push directly to `main`; open a pull request.
2. Branches must be up to date with `main` before merging.
3. Each PR must be reviewed by another team member.
4. Fill out the PR template before requesting review.
5. PR titles must start with the author name, for example:
   `Igor: Add reproducible evaluation and reporting`.
6. Each PR must pass metadata, code-quality, notebook, and reproducibility
   checks.

## Authors

- Paula Banach, 440186
- Igor Kolodziej, 440239
- Mikita Silivestrau, 392905

## Work Split

### Paula

1. Write the data ingestion script.
2. Build the Exploratory Data Analysis notebook.
3. Write preprocessing that encodes categorical variables and prepares model
   features.
4. Train a linear regression baseline.
5. Maintain repository rules and GitHub Actions.

### Mikita

1. Build non-linear models on the preprocessed data.
2. Write reproducible hyperparameter tuning.
3. Centralize random seeds and split settings.
4. Save the final trained model as an artifact.
5. Compare model families and select the final model.

### Igor

1. Manage the Poetry environment and lock file.
2. Write final held-out evaluation for the saved model.
3. Generate model explainability charts.
4. Maintain the master README with end-to-end instructions.
5. Prepare the final report comparing project results with previous work.
