# House Price Prediction — ML Experiment Pipeline

A structured machine learning project comparing multiple feature engineering strategies for predicting house prices using Linear Regression and Ridge Regression. Built on the Kaggle House Prices dataset with 1,454 real entries.

## What Makes This Different

This is not a single model trained on one feature set. It is a controlled experiment pipeline that tests 3 different feature configurations, compares them using both test R² and 5-fold cross-validation, and selects the best model with clear reasoning.

## Dataset

Kaggle House Prices — Advanced Regression Techniques
1,454 rows · Features used: living area, bedrooms above ground, house age

## Experiments

| Experiment | Features | Test R² | CV Mean R² | CV Std |
|---|---|---|---|---|
| Baseline | area, bedrooms, age | 0.7165 | 0.6731 | ±0.0589 |
| + area_per_bedroom | above + engineered feature | 0.7170 | 0.6706 | ±0.0579 |
| + log_area | above + log transform | 0.7065 | 0.6639 | ±0.0642 |

Experiment 2 was selected as the final model. The engineered feature area_per_bedroom gave a marginal improvement while maintaining simplicity. log_area was tested and removed as it reduced both test and cross-validation performance.

## Key Findings

- Area is the strongest positive predictor with a coefficient of +56,267
- Bedrooms has a negative coefficient of -12,599, meaning more bedrooms without proportionally more area implies smaller rooms and lower price
- Cross-validation scores are consistent across all folds, confirming the model generalizes well and is not overfitting

## Final Model Performance

- Algorithm: Linear Regression
- Features: area, bedrooms, age, area_per_bedroom
- Test R²: 0.7170
- Test MSE: 1,515,285,517
- CV Mean R²: 0.6706 ± 0.0579

## Feature Coefficients

| Feature | Coefficient |
|---|---|
| area | +56,267.66 |
| area_per_bedroom | +1,742.75 |
| bedrooms | -12,599.48 |
| age | -29,506.12 |
| Intercept | 183,954.03 |

## Project Structure

```
├── data/                  # Dataset files
├── model/                 # Saved model and scaler (joblib)
├── notebooks/             # Jupyter notebooks with full experiment pipeline
└── requirements.txt
```

## Tech Stack

- Python
- scikit-learn
- Pandas
- NumPy
- Matplotlib
- joblib

## How to Run

```bash
git clone https://github.com/Ragib-Waquar/House-price-prediction-model.git
cd House-price-prediction-model
pip install -r requirements.txt
jupyter notebook notebooks/
```

## Author

Ragib Waquar — BTech CSE Student
