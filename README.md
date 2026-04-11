# Credit Card Fraud Detection API

This project is a complete machine learning pipeline built to detect fraudulent credit card transactions, along with a deployed API to serve predictions in real-time.

The goal was not just to train a model, but to understand the problem deeply and build something end-to-end — from data preprocessing to deployment.

---

## Problem Context

Credit card fraud detection is a classic **imbalanced classification problem**.

- Fraud cases are extremely rare (~0.17%)
- Most transactions are legitimate
- This makes accuracy a misleading metric

The dataset used is publicly available on Kaggle:
**mlg-ulb/creditcardfraud**

Important note:
- Features `V1–V28` are anonymized (PCA transformed)
- Only `Amount` and `Time` are interpretable

---

## Approach

### 1. Data Preprocessing
- Removed duplicates
- Checked for missing values
- Performed basic statistical analysis
- Observed extreme class imbalance

---

### 2. Exploratory Data Analysis (EDA)
- Visualized class distribution
- Analyzed transaction amounts vs fraud
- Checked time-based patterns
- Generated correlation heatmap

Key insight:
> Fraud detection requires focusing on minority class behavior, not overall accuracy.

---

### 3. Handling Class Imbalance

Used **SMOTE (Synthetic Minority Oversampling Technique)**

Why?
- Original fraud samples were too few
- Model would otherwise ignore fraud cases
- SMOTE helps balance the dataset artificially

---

### 4. Feature Scaling

- Scaled `Amount` and `Time` using `StandardScaler`
- Other features already transformed (PCA)

---

### 5. Model Training

Trained multiple models instead of jumping to one:

- Logistic Regression → baseline
- Random Forest → main model
- ExtraTrees + HistGradientBoosting → additional comparison

Why multiple models?
> To compare performance and understand trade-offs instead of blindly choosing one.

---

### 6. Model Evaluation

Used:
- Precision
- Recall
- F1-score (especially for fraud class)

Why not accuracy?
> Because predicting "not fraud" always would still give ~99% accuracy.

---

### 7. Threshold Tuning

Instead of relying on default 0.5:

- Tested multiple thresholds (0.5 → 0.95)

Why?
- Fraud detection requires balancing:
  - catching fraud (recall)
  - avoiding false alarms (precision)

---

### 8. Final Model Selection

**Random Forest was selected**

Reason:
- Best balance between precision and recall
- Strong performance on minority class
- Fewer false positives compared to others

---

## API Design

Built using **FastAPI** and deployed on Render.

### Live API:
https://credit-card-fraud-detection-api-bx3x.onrender.com

---

## How Prediction Works

User provides:
```json
{
  "amount": 50,
  "time": 10000
}
```

---

## Important Design Decision

### Why only Amount and Time as input?

Because:

- Other features are anonymized (PCA)
- Their real-world meaning is unknown
- Asking users for them is not practical

So:

> Default (mean) values are used for remaining features

This is a practical compromise, not a perfect solution.

---

## Running Locally

```bash
git clone https://github.com/rohitnath-dev/credit-card-fraud-detection-api.git
cd credit-card-fraud-detection-api

pip install -r requirements.txt

uvicorn app.main:app --reload
```

## API Documentation

Open the API docs here:  
http://127.0.0.1:8000/docs

---

## Limitations

- Dataset is anonymized → limited interpretability  
- Input features are incomplete (only 2 real inputs)  
- Model is not trained on real-world production data  
- SMOTE creates synthetic samples (not real fraud cases)  


---

## Future Improvements

- Use real-world transaction features  
- Better feature engineering  
- Advanced models (XGBoost, LightGBM)  
- Real-time streaming system  
- Frontend integration  


---

## Final Note

This project focuses on understanding the **full ML pipeline**, not just model training.

data → model → evaluation → API → deployment

Everything is built and connected end-to-end.
