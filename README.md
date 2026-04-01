# Medical Insurance Cost Prediction
This repository is a final assignment for the Reproducible Research course at the University of Warsaw.

## Project Goal
Build a reproducible Machine Learning pipeline that predicts the medical insurance costs for an individual based on their demographic and lifestyle factors.

## Dataset 
We will use the widely trusted [Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) from Kaggle.

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

### Igor
1. Manage the Python environment. Create and test the requirements.txt or environment.yml file to ensure everyone can install the exact same library versions.
2. Write an evaluation script (src/evaluate.py) that calculates final metrics on the test set.
3. Implement model explainability to automatically generate charts showing why the model made its predictions.
4. Write the master README.md with step-by-step instructions on how to clone the repo, install dependencies, and run the pipeline from start to finish.


