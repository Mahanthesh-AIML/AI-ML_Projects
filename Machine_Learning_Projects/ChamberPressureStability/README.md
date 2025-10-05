# Chamber Pressure Stability – Machine Learning Classification

This project builds a machine learning pipeline to predict **chamber pressure stability** in semiconductor processing systems.  
It uses sensor data (gas flows, pump performance, and valve states) to identify whether the chamber remains **stable (1)** or becomes **unstable (0)** during operation.

The notebook performs full data preprocessing, feature engineering, modeling, and explainability using SHAP.

---

## 1. Objective

Pressure stability is critical in semiconductor etching and deposition.  
Even a small deviation (±2 mTorr) can lead to:
- Wafer non-uniformity and reduced yield  
- Process drift over time  
- Increased equipment downtime  

**Goal:**  
Predict if the chamber run will remain *Stable* or become *Unstable* using recipe and sensor features.

---

## 2. Dataset

- **Rows:** 50,500 (synthetic but modeled after real fab chamber logs)  
- **Features:**
  - Gas flows: `CF4_set`, `O2_set`, `Ar_set`, and their actuals  
  - Pump metrics: `Pump_RPM`, `Pump_Power`  
  - Exhaust: `Valve_Position`  
  - Chamber: `Base_Pressure`  
  - Derived: flow errors, ratios (`O2_CF4_ratio`, `Ar_CF4_ratio`), `pressure_delta`, and `instability_index`
- **Target:**  
  `Stable` → 1 (stable), 0 (unstable)

Data file: `data/chamber_pressure_stability_check.csv`

---

## 3. Data Preprocessing

| Step | Description |
|------|--------------|
| **Missing values** | Imputed with column median |
| **Duplicates** | Removed using `drop_duplicates()` |
| **Outlier handling** | Applied IQR clipping on `Pump_RPM` and `Valve_Position` |
| **Normalization** | Computed ratios and normalized valve positions |
| **Encoding** | One-hot encoded `flow_zone` and `valve_zone` bins |

---

## 4. Feature Engineering

- **Gas Flow Features**
  - Total flow: `CF4_actual + O2_actual + Ar_actual`
  - Flow balance (sum of absolute flow errors)
  - Ratios: `O2_CF4_ratio`, `Ar_CF4_ratio`

- **Pump & Exhaust Features**
  - Efficiency = `Pump_RPM / Pump_Power`
  - Load factor = `Pump_RPM × (1 - valve_normalized)`

- **Chamber Condition Features**
  - Pressure setpoint = total set flow ÷ pump RPM
  - Pressure delta = setpoint − base pressure
  - Instability index = combined flow imbalance × gas ratio ÷ pump efficiency

---

## 5. Modeling and Evaluation

### Algorithms Tested
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- KNN  
- XGBoost  
- CatBoost  

### Evaluation Metrics
- **Accuracy**
- **Precision / Recall / F1-Score**
- **ROC-AUC**

### Class Imbalance Handling
Used **SMOTE (Synthetic Minority Over-Sampling Technique)** on the training set.

### Model Results (Top Performers)

| Model | Accuracy | F1 | ROC-AUC |
|--------|-----------|----|---------|
| Gradient Boosting | ~0.73 | **0.73** | **0.59** |
| Random Forest | ~0.70 | 0.68 | 0.56 |
| XGBoost | ~0.69 | 0.66 | 0.55 |

✅ **Gradient Boosting** was selected as the final model due to the best F1 and stability tradeoff.

---

## 6. Explainability (SHAP)

SHAP (SHapley Additive Explanations) was used for global and local interpretability.

### Key Insights
- High `O2_CF4_ratio` and closed valve positions correlate with instability.  
- Pump inefficiency and large flow imbalance contribute to drift.  
- The SHAP summary plot aligns with domain intuition on chamber dynamics.

---

## 7. Output Artifacts

| File | Description |
|------|--------------|
| `gradientboosting_chamber_stability.pkl` | Saved trained model |
| `results_df.csv` | Model comparison metrics |
| SHAP plots | Global and local feature impact visualizations |

---

## 8. Tools & Libraries

| Category | Tools |
|-----------|--------|
| Environment | Python 3.10, Google Colab |
| Libraries | pandas, numpy, seaborn, matplotlib, scikit-learn, imblearn, xgboost, catboost, shap |
| Storage | joblib for model saving |

---

## 9. Future Improvements

- Hyperparameter tuning for ensemble models  
- Real-time inference on streaming sensor data  
- Additional features: chamber temperature, gas feedback control response  
- Integration with process fault detection pipelines

---

## 10. Project Structure

