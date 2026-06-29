# Advance-Statistics
# Advanced Statistical Analysis & Modeling

**Author:** Abdul Rafay Khan
**Institution:** Ghulam Ishaq Khan Institute of Engineering Sciences and Technology (GIKI)

## 🚀 Project Overview
This repository contains a comprehensive suite of Python implementations focused on core Advanced Statistics and machine learning methodologies. It bridges the gap between raw data analysis, feature engineering, and robust statistical modeling using the CDC BRFSS health indicators dataset.

## 🛠️ Analyzed Methods & Scripts

### 1. Exploratory Data Analysis & Correlations (`01_eda_and_correlations.py`)
A comparative script that performs data cleaning, variable distribution analysis, and correlation mapping.
* **Descriptive Statistics:** Measures of central tendency and dispersion.
* **Visualizations:** Distribution histograms and correlation heatmaps (Pearson r).
* **Outlier Detection:** Range validation for continuous and ordinal variables.

### 2. Feature Engineering (`02_feature_engineering.py`)
Pipeline for transforming raw data into model-ready features.
* **Polynomial Transformations:** Capturing nonlinear effects in age and general health.
* **Log-Transformations:** Stabilizing variance in zero-inflated count variables.
* **Interaction Terms:** Capturing compounding risks (e.g., Blood Pressure × Cholesterol).
* **Standardization:** Z-score scaling for comparative coefficient interpretation.

### 3. Linear Regression (`03_linear_regression.py`)
Implementation of OLS (Ordinary Least Squares) models.
* **Simple Linear Regression:** BMI prediction based on general health.
* **Multiple Linear Regression:** 16+ predictor models with diagnostic checks.
* **Diagnostics:** Residual analysis, Normal Q-Q plots, and Breusch-Pagan homoscedasticity tests.

### 4. Generalized Linear Models (`04_glm_modeling.py`)
GLM implementations for non-normal response variables.
* **Logistic Regression:** Binary classification for diabetes risk with Odds Ratio analysis.
* **Poisson Regression:** Modeling count responses (physically unhealthy days) with overdispersion testing.

### 5. Nonlinear Modeling (`05_nonlinear_modeling.py`)
Comparison of regression strategies for BMI-age dynamics.
* **Polynomial Regression:** Fitting cubic trends to age-BMI relationships.
* **NLS Logistic Saturation:** Modeling BMI as a saturation function (amplitude, rate, inflection point).
* **NLS Exponential Growth:** Testing unbounded exponential growth hypothesis.

### 6. Model Comparison (`06_model_comparison.py`)
Critical evaluation of model performance.
* **Information Criteria:** Comparative AIC and BIC analysis for model selection.
* **Bias-Variance Tradeoff:** Train/Test R² gap analysis to detect overfitting.
* **Interpretability vs. Flexibility:** Mapping models to find the ideal balance for prediction.

## 💻 How to Run
1. Clone this repository to your local machine.
2. Install the necessary mathematical visualization libraries:
```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn
