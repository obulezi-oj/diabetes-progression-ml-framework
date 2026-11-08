# Demographic Equity, Explainability, and Clinical Net Benefit of Machine Learning Ensembles in Diabetes Risk Stratification: A 5-Fold Validated Cohort Analysis

[![R Version](https://img.shields.io/badge/R-v4.3.2%2B-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Cohort Size](https://img.shields.io/badge/Cohort-N%20%3D%20100%2C000-green.svg)]()

## 📋 Overview

This repository contains the full production-ready **R** analytical pipeline for predictive modeling, demographic equity assessment, and explainable AI in diabetes progression. Designed to meet the reporting standards of Q1 medical informatics and digital health journals (e.g., *BioData Mining*, *JAMIA*, *IEEE JBHI*), the framework evaluates **10 machine learning and deep learning architectures** across a large cohort ($N = 100,000$).

### Key Features
* **Comprehensive ML/DL Benchmarking:** 10 algorithms including XGBoost, DART Boosting, Random Forest, Multi-Layer Perceptron (MLP), Naive Bayes, Decision Trees, and Multivariable Logistic Regression.
* **Rigorous Validation:** 5-fold stratified cross-validation with empirical 95% Confidence Intervals ($\text{Mean} \pm 1.96 \cdot \text{SE}$) for Accuracy, Sensitivity, Specificity, F1-Score, AUC-ROC, and AUPRC.
* **Epidemiological Risk Quantification:** Multivariable Logistic Regression calculating Adjusted Odds Ratios ($\text{aOR}$) with 95% CIs and exact $z$-scores.
* **Multi-Class Disease Staging:** Non-binary progression modeling across 5 clinical stages (*No Diabetes*, *Pre-Diabetes*, *Type 2*, *Type 1*, and *Gestational Diabetes*).
* **Demographic Equity & Fairness:** Algorithmic bias validation stratified across Age Brackets ($<40$, $40\text{--}64$, $\ge 65$), Gender identities, and Ethnic cohorts.
* **Explainable AI (XAI):** Game-theoretic feature attributions via SHapley Additive exPlanations (SHAP) and Partial Dependence Plots (PDP).
* **Clinical Utility & Reliability:** Probability calibration (Brier Scores) and Decision Curve Analysis (DCA) measuring Net Clinical Benefit across threshold probabilities ($p_t = 0.10 \text{--} 0.90$).

---

## 🗂️ Output Artifacts Generated

Running the master pipeline automatically outputs **8 publication-ready CSV tables** and **10 high-resolution PNG figures** (300 DPI) into `results_output_ultimate_publication/`:

### Tables (.csv)
1. `Table1_Cohort_Demographics_Baseline.csv` - Baseline epidemiological characteristics ($N = 100,000$).
2. `Table2_Binary_Classification_10Models_5FoldCV.csv` - 5-Fold CV metrics with empirical 95% CIs across all models.
3. `Table3_Multivariable_Logistic_Regression_aOR.csv` - Multivariable Adjusted Odds Ratios ($\text{aOR}$) and $p$-values.
4. `Table4_Multiclass_Stage_Progression_Metrics.csv` - Multi-class sensitivity, specificity, PPV, NPV, and F1-scores.
5. `Table5_Subgroup_Fairness_Gender_Age_Ethnicity.csv` - Stratified fairness metrics across demographic cohorts.
6. `Table6_Feature_Permutation_Importance_Ranking.csv` - Feature Gain, Cover, and Frequency rankings.
7. `Table7_Model_Calibration_and_Brier_Scores.csv` - Brier Scores, calibration slopes, and intercepts.
8. `Table8_Decision_Curve_Net_Benefit_Analysis.csv` - DCA Net Benefit values across decision thresholds.

### Figures (.png)
* `Fig1_Correlation_Matrix.png` - Clinical biomarker correlation matrix heatmap.
* `Fig2_Biomarker_Progression.png` - HbA1c and Fasting Glucose boxplots across progression stages.
* `Fig3_ROC_Curves_Comparison.png` - Comparative ROC overlay curves.
* `Fig4_Confusion_Matrix.png` - Normalized confusion matrix heatmap for XGBoost.
* `Fig5_SHAP_Feature_Importance.png` - SHAP global feature contribution beeswarm plot.
* `Fig6_Partial_Dependence_Plots.png` - Non-linear PDP curves for HbA1c and Fasting Glucose thresholds.
* `Fig7_Calibration_Curve.png` - Model calibration reliability curve against the $45^\circ$ ideal.
* `Fig8_Decision_Curve_Analysis.png` - Net clinical benefit curves comparing XGBoost vs. Treat All / Treat None.
* `Fig9_Subgroup_Equity.png` - Subgroup discrimination (AUC) across Age, Ethnicity, and Gender.
* `Fig10_Deep_Insight_Pathophysiology.png` - 4-panel pathophysiological exploration plot.

---

## 🛠️ Installation & Setup

### Prerequisites
* **R (v4.0.0 or higher)**
* **RStudio** (recommended)

### Required R Packages
The script automatically detects and installs missing packages, but you can manually install them via:

```R
install.packages(c(
  "tidyverse", "caret", "rpart", "randomForest", 
  "xgboost", "pROC", "shapviz", "gridExtra", 
  "e1071", "nnet", "PRROC", "viridis"
))
