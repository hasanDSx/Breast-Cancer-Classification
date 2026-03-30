# 🎗️ Breast Cancer Classification — SVM + PCA Pipeline

> **Clinical principle:** Missing a real cancer case is far more costly than a false alarm.  
> → **Recall on the malignant class** is the primary optimization metric — not accuracy.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Key Takeaway](#key-takeaway)

---

## Project Overview

A complete Machine Learning pipeline that classifies breast tumors as **Malignant** or **Benign** using the Wisconsin Breast Cancer Dataset (569 patients, 30 features).

The pipeline is built around a single clinical principle: **missing a real cancer case is far more costly than a false alarm.** For that reason, Recall on the malignant class was chosen as the primary optimization metric — not overall accuracy.

---

## Dataset

| Field | Value |
|---|---|
| **Source** | Wisconsin Breast Cancer (UCI) |
| **Patients** | 569 (357 Benign · 212 Malignant) |
| **Features** | 30 numerical measurements per tumor |
| **Target** | Malignant (1) · Benign (0) |
| **Primary metric** | Recall (malignant class) |
| **Train / Test split** | 80% / 20% — stratified |

---

## Methodology

### 1. Exploratory Data Analysis

- Confirmed zero missing values and no duplicate rows
- Class distribution: **62.7% Benign, 37.3% Malignant** — slight imbalance, reinforcing the choice of Recall over Accuracy
- Violin plots revealed that `concave_points_worst`, `area_worst`, and `perimeter_worst` showed the strongest class separation
- Full correlation heatmap identified high inter-feature correlation → motivating PCA for dimensionality reduction
- Top 10 features most correlated with diagnosis ranked and visualized

### 2. Preprocessing

| Step | Detail |
|---|---|
| **Feature scaling** | `StandardScaler` — SVM and PCA are both sensitive to feature scale |
| **Dimensionality reduction** | PCA after scaling: 30 features → 10 components capturing ~95% of total variance |
| **Train/test split** | 455 training / 114 test patients, stratified by class |

### 3. Pipeline Architecture

A single `scikit-learn` Pipeline chains all steps to **prevent data leakage** — scaling and PCA are fitted only on the training fold during cross-validation, never on the validation fold.

```
StandardScaler → PCA (10 components) → SVC (RBF kernel)
```

| Step | Component | Purpose |
|---|---|---|
| 1 | `StandardScaler` | Normalizes all features to mean=0, std=1 |
| 2 | `PCA` (or passthrough) | Reduces dimensionality — also tested without PCA |
| 3 | `SVC` / `LogisticRegression` | Classifier selected by GridSearchCV |

### 4. Hyperparameter Optimization

`GridSearchCV` tested **57 candidate configurations** across **5 stratified folds** (285 total fits), optimizing for Recall.

| Parameter | Best Value |
|---|---|
| **CV strategy** | Stratified K-Fold (5 splits) |
| **Optimization metric** | Recall (malignant class) |
| **Best classifier** | SVC — RBF kernel |
| **Best C** | 10 |
| **Best gamma** | `scale` |
| **PCA components** | 10 |
| **Best CV Recall score** | **0.96** |

---

## Results

### Key Metrics — Test Set

> All metrics computed on the held-out test set (114 patients never seen during training).

| Metric | Value | Meaning |
|---|---|---|
| ✅ **Accuracy** | **96.5%** | Only 4 of 114 unseen patients misclassified |
| 🎯 **Recall** | **92.9%** | Caught 39 out of 42 real cancer cases ← *primary target* |
| ⚖️ **Precision** | **97.5%** | 97.5% of predicted malignant cases were actually malignant |
| 📊 **F1-Score** | **95.1%** | Harmonic mean of Precision & Recall |
| 📈 **AUC-ROC** | **0.993** | Near-perfect class separation (1.0 = perfect, 0.5 = random) |

### Train vs. Test Comparison

The gap between train and test performance is small — the model generalizes well without significant overfitting.

| Split | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| **Train (455)** | 98.9% | 100% | **97.1%** | 98.5% | — |
| **Test (114)** | 96.5% | 97.5% | **92.9%** | 95.1% | 0.993 |

### Per-Class Report — Test Set

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Benign (0)** | 95.95% | 98.61% | 97.26% | 72 |
| **Malignant (1)** | 97.50% | **92.86%** | 95.12% | 42 |
| **Weighted avg** | 96.52% | 96.49% | 96.47% | 114 |

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **ML** | Scikit-Learn — `Pipeline`, `SVC`, `LogisticRegression`, `GridSearchCV`, `StratifiedKFold`, `PCA` |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn — violin plots, heatmaps, confusion matrices, ROC & PR curves |
| **Metrics** | Accuracy, Recall, Precision, F1-Score, AUC-ROC, Average Precision |

---

## Key Takeaway

> *In oncology, the metric you choose is the decision you are making.*  
> **Recall was the North Star of this project** — because in cancer diagnosis, missing even one case is one too many.

**92.9% Recall = 39 of 42 actual cancer cases caught.**

---

*Full code, notebook, and visualizations are available in the repository.*
