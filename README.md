# Medical Insurance Cost Prediction
This repository is a final assignment for the Reproducible Research course at the University of Warsaw.

## Project Goal
Build a reproducible Machine Learning pipeline that predicts the medical insurance costs for an individual based on their demographic and lifestyle factors.

## Dataset 
We will use the widely trusted [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) from Kaggle.

### Downloading the data

1. Get your Kaggle API credentials: go to [Kaggle](https://www.kaggle.com) → **Settings → API → Generate New Token**. Copy the `username` and `key` values shown.
2. Open create a `kaggle.json` file in the `.kaggle` folder at the root of this repo and fill in your credentials:
   ```json
   {"username": "your_kaggle_username", "key": "your_kaggle_api_key"}
   ```
3. Create and activate a virtual environment, then install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS / Linux
   pip install -r requirements.txt
   ```
4. Run the ingestion script:
   ```bash
   python scripts/ingest_data.py
   ```
   The raw dataset will be saved to `data/insurance.csv`.

### Exploratory Data Analysis

The notebook [`notebooks/eda.ipynb`](notebooks/eda.ipynb) provides a full walkthrough of the dataset before any modelling. It covers:

- **Data quality**: dtype inspection, missing values, and duplicate checks
- **Target variable**: distribution and log-transform of `charges`
- **Numeric features**: histograms and boxplots for `age`, `bmi`, and `children`
- **Categorical features**: count plots for `sex`, `smoker`, `region`, and `children`
- **Charges by group**: boxplots and mean charges broken down by categorical variables
- **Charges vs numeric features**: scatter plots coloured by smoking status
- **Correlation analysis**: heatmap on label-encoded features

All figures are saved to `reports/figures/`. Running the final cell exports the notebook as a self-contained HTML report to `reports/eda_report.html`.

## IMPORTANT: Branch and PR Rules
When creating a new branch, please start with your name and specify which task you will be doing. Example: `paula/first-readme-update`
The repository is being protected with a set of rules to ensure smooth collaboration:
1. You cannot push directly onto main. You need to make a PR with your changes.
2. Branches need to be up-to-date with main before merging.
3. Each PR needs to be reviewed by another Team member.
4. The PR description will be auto-populated by a template defined in `.github/pull_request_template.md`. Please fill it out before requesting reviews.
5. The PR title needs to start with the author name. Example: *"Paula: Add PR healthchecks"*
6. Each PR needs to pass the dedicated healthchecks:
   - `.github/workflows/pr-metadata.yml`: A workflow that enforces the PR naming convention (Paula:, Mikita:, or Igor:) and ensures this PR body is not empty.
   - `.github/workflows/code-quality.yml`: A workflow that automatically checks our Python formatting (black, isort), lints for errors (flake8), and rejects the PR if unstripped Jupyter Notebook outputs are detected (nbstripout).
   - `.github/workflows/reproducibility.yml`: A workflow that verifies our requirements.txt is up-to-date by attempting a fresh install on an Ubuntu runner.

## Authors
- Paula Banach, *440186*
- Igor Kołodziej, *440239*
- Mikita Silivestrau, *392905*

## Work-Split

### Paula
1. Write the data ingestion script.
2. Build the Exploratory Data Analysis (EDA) script to visualize distributions.
3. Write a preprocessing script that encodes categorical variables (sex, smoker, region) and scales numerical ones (BMI, age).
4. Train a simple Linear Regression model to establish a baseline error metric.
5. Maintain the repository, add branch rules, GH actions.

### Mikita
1. Build non-linear models on the preprocessed data.
2. Write the script for hyperparameter tuning.
3. Ensure strict reproducibility by setting global random seeds for all model training and train/test splits.
4. Save the final trained model as an artifact.
5. Compare different models to see which one performs best.

### Igor
1. Manage the Python environment. Create and test the requirements.txt or environment.yml file to ensure everyone can install the exact same library versions.
2. Write an evaluation script that calculates final metrics on the test set.
3. Implement model explainability to automatically generate charts showing why the model made its predictions.
4. Write the master README.md with step-by-step instructions on how to clone the repo, install dependencies, and run the pipeline from start to finish.
5. Prepare the report, comparing the previous results from Kaggle users to our final results.


