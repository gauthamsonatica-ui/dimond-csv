# 💎 Diamond Dynamics: Price Prediction & Market Segmentation

**An end-to-end machine learning system for predicting diamond prices and segmenting the diamond market using regression, deep learning (ANN), clustering, and an interactive Streamlit application.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![scikit--learn](https://img.shields.io/badge/ML-scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/DL-TensorFlow%2FKeras-FF6F00.svg)](https://www.tensorflow.org/)

---

## 📌 Overview

The diamond market's pricing is driven by a complex interplay of quality attributes — **carat, cut, color, and clarity** — commonly known as the "4Cs." This project builds a complete analytics pipeline that:

1. **Predicts diamond prices** using multiple regression algorithms and an Artificial Neural Network (ANN)
2. **Segments diamonds into meaningful market clusters** (e.g., *Premium Heavy*, *Mid-range Balanced*, *Affordable Small*) using unsupervised learning
3. **Serves both capabilities through an interactive Streamlit web application** for real-time price prediction and market segment identification

This enables retailers to build **dynamic pricing strategies**, manage **inventory categorization**, power **luxury product recommendation engines**, and support **customer segmentation** for personalized marketing.

---

## 🎯 Objectives

- Predict diamond prices using multiple ML regression models and an ANN
- Cluster diamonds into market segments based on physical and qualitative features
- Deliver an interactive Streamlit UI for real-time price prediction and cluster identification based on user inputs

---

## 🌍 Domain

**E-Commerce · Luxury Goods Analytics · Retail Pricing Optimization**

---

## 🛠️ Skills Demonstrated

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA) & Visualization
- Feature Engineering
- Outlier & Skewness Handling
- Regression Modeling (Classical ML + ANN)
- K-Means Clustering & Cluster Labeling
- Dimensionality Reduction with PCA
- Streamlit UI Design & Model Deployment

---

## 📊 Dataset

| Detail | Description |
|---|---|
| **Source** | Diamond dataset (classic diamonds pricing dataset) |
| **Shape** | 53,940 rows × 10 features |

### Column Descriptions

| Column | Description |
|---|---|
| `carat` | Weight of the diamond (in carats) — primary price driver |
| `cut` | Cut quality — ordinal: *Fair, Good, Very Good, Premium, Ideal* |
| `color` | Color grade from best to worst — ordinal: *D, E, F, G, H, I, J* |
| `clarity` | Inclusion/blemish grade: *IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1* |
| `depth` | Total depth % = `(z / mean(x, y)) × 100` |
| `table` | Width of the top facet as % of average diameter |
| `price` | Target variable — price in USD (converted to INR in this project) |
| `x`, `y`, `z` | Length, width, and depth of the diamond in mm |

---

## 🏗️ Pipeline Architecture

```
Raw Diamond Dataset
        │
        ▼
Data Cleaning (zero/invalid x,y,z → NaN → drop/impute)
        │
        ▼
Feature Engineering
  ├── price_inr        = price × conversion rate
  ├── volume            = x × y × z
  ├── price_per_carat   = price / carat
  ├── dimension_ratio   = (x + y) / (2 × z)
  └── carat_category    = Light / Medium / Heavy
        │
        ▼
Outlier & Skewness Handling (IQR / Z-score, log / sqrt / box-cox)
        │
        ▼
Encoding (Label / Ordinal Encoding for cut, color, clarity, carat_category)
        │
   ┌────┴────┐
   ▼         ▼
Regression   Clustering
(Price       (Market
Prediction)  Segmentation)
   │         │
   ▼         ▼
Model Comparison   K-Means + PCA
(LR, DT, RF, ANN)  (Elbow / Silhouette)
   │         │
   └────┬────┘
        ▼
  Best Models Pickled (.pkl)
        │
        ▼
  Streamlit Web Application
  (EDA · Model Comparison · Prediction)
```

---

## 🤖 Model Building

### 1️⃣ Regression — Price Prediction
- Train-test split (80–20)
- Models: **Linear Regression, Decision Tree, Random Forest**, and an **ANN** (Keras/TensorFlow, `Dense(64) → Dense(32) → Dense(1)`)
- Evaluated using **MAE, MSE, RMSE, R²**
- Best-performing model comparison via bar chart, saved as `.pkl`

### 2️⃣ Clustering — Market Segmentation
- **K-Means clustering** (n=3), with optional DBSCAN / Hierarchical clustering
- Optimal cluster count via **Elbow Method** or **Silhouette Score**
- Features standardized (`StandardScaler`) before fitting
- **PCA** for dimensionality reduction and 2D cluster visualization
- Clusters labeled by analyzing average price, carat, and cut per group:

| Cluster Profile | Label |
|---|---|
| High carat, high price | **Premium Heavy Diamonds** |
| Low carat, low price | **Affordable Small Diamonds** |
| Medium carat, medium price | **Mid-range Balanced Diamonds** |

---

## 📱 Streamlit Application

The app (`app.py`) provides three sections:

- **EDA** — Distribution plots, boxplots, correlation heatmap, regression plots
- **Model Comparison** — Side-by-side R² and RMSE across all trained models
- **Prediction** — Interactive form to input diamond attributes and get:
  - 💰 Predicted price (in INR)
  - 🧭 Predicted market cluster with its label
  - 📈 PCA-based cluster visualization

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Repository Structure

```
Diamond-Dynamics-Price-Prediction-Market-Segmentation/
├── diamond.ipynb / diamond.py     # Main notebook — EDA, modeling, app generation
├── app.py                          # Streamlit web application
├── models/
│   ├── regression_model.pkl        # Best regression model
│   └── clustering_model.pkl        # Best clustering model
├── data/                           # Dataset (or download link)
├── requirements.txt
└── README.md
```

---

## 📈 Evaluation Metrics

| Task | Metrics |
|---|---|
| **Regression** | MAE, MSE, RMSE, R² |
| **Clustering** | Inertia, Silhouette Score |

---

## 🏷️ Technical Tags

`machine learning` `regression` `artificial neural network` `random forest` `decision tree` `xgboost` `knn` `model evaluation` `feature engineering` `feature selection` `feature importance` `outlier detection` `skewness treatment` `clustering` `k-means` `elbow method` `silhouette score` `streamlit` `model deployment` `diamond price prediction` `market segmentation` `scikit-learn` `tensorflow` `keras` `eda` `matplotlib` `seaborn` `pkl model saving`

---

## 🚀 Real-World Use Cases

- **Dynamic pricing strategy** for diamond retailers
- **Inventory categorization** and market segmentation for product listings
- **Luxury product recommendation engines** based on diamond profiles
- **Customer segmentation** for personalized marketing

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
git clone https://github.com/<your-username>/Diamond-Dynamics-Price-Prediction-Market-Segmentation.git
cd Diamond-Dynamics-Price-Prediction-Market-Segmentation
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app.py
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

Project developed as part of the **GUVI × HCL "Skill Up. Level Up"** Data Science / AI-ML capstone program.
