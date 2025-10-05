# Tool Wear Prediction – Machine Learning Project

This project builds and benchmarks multiple machine learning classifiers to predict **CNC tool wear state (Usable vs Worn)** from machining sensor data.

---

## 1. Objective

Tool wear affects product quality, machining accuracy, and production cost.  
Instead of relying on fixed maintenance intervals, this project uses sensor data to **predict worn tools early**, reducing downtime and scrap.

---

## 2. Dataset Overview

| Type | Description |
|------|--------------|
| Operating | Spindle speed, feed rate, depth of cut |
| Sensor | Load, vibration, temperature |
| Derived | stress_factor, thermo_mech_index, temp_slope, load_variation |
| Target | `tool_worn` → 1 (worn), 0 (usable) |

Dataset size: ~20k–50k samples (varies by machine log)

---

## 3. Feature Engineering

| Feature | Description |
|----------|-------------|
| **hours_cumulative** | Tool usage duration |
| **cycle_index** | Machining cycle number |
| **temp_slope** | Rate of temperature rise per cycle |
| **load_variation** | Rolling std of spindle load |
| **vibration_variation** | Rolling std of vibration |
| **stress_factor** | `spindle_load × hours_run` |
| **thermo_mech_index** | `spindle_load × temperature` |

Each feature captures a physical or mechanical aspect of tool degradation.

---

## 4. Models Benchmarked

| Model Type | Notes |
|-------------|-------|
| Logistic Regression | Interpretable baseline |
| Decision Tree | Captures nonlinear thresholds |
| Random Forest | Ensemble of trees for robustness |
| Gradient Boosting | Sequential boosting for accuracy |
| XGBoost | Gradient-boosted trees with regularization |
| CatBoost | Handles categorical and noisy features efficiently |

Cross-validation: **5-Fold Stratified**

Metrics:
- AUC  
- Recall (Worn tools)  
- Precision (Worn tools)  
- F1-Score  
- Accuracy

---

## 5. Results Summary

| Model | AUC | Recall | Precision | F1 | Accuracy |
|--------|------|---------|------------|----|-----------|
| **XGBoost** | 0.74 | 0.67 | 0.66 | **0.66** | 0.72 |
| CatBoost | 0.73 | **0.68** | 0.65 | **0.66** | 0.71 |
| Gradient Boosting | 0.72 | 0.64 | 0.64 | 0.64 | 0.70 |
| Random Forest | 0.69 | 0.61 | 0.59 | 0.60 | 0.69 |
| Logistic Regression | 0.61 | 0.48 | 0.52 | 0.50 | 0.67 |
| Decision Tree | 0.58 | 0.46 | 0.51 | 0.48 | 0.66 |

**Best model:** XGBoost → Strongest recall and F1-score for worn tools.

---

## 6. Visualizations

- Confusion matrices and ROC curves for all models  
- Model comparison bar chart for Accuracy, Recall, Precision, F1, and AUC  

---

## 7. Output Artifacts

| File | Description |
|------|--------------|
| `best_tool_wear_model.pkl` | Trained XGBoost model |
| `results.csv` | Summary of model metrics |
| `plots/` | All confusion and ROC plots |

---

## 8. Tech Stack

| Category | Tools |
|-----------|--------|
| Language | Python 3.10 |
| Libraries | pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost, catboost |
| Environment | Google Colab / Jupyter |
| Model Storage | joblib |

---

## 9. Insights

- Wear progression is best captured by combined **thermal + mechanical** indices.  
- **XGBoost** handles nonlinearity and imbalance better than linear models.  
- High recall for worn tools ensures safer maintenance planning.

---

## 10. Next Steps

- Hyperparameter tuning for top models  
- Feature importance and SHAP analysis  
- Real-time inference integration in a monitoring dashboard  

---

### 🧑‍💻 Author
**<Your Name>**  
Machine Learning Engineer | Applied AI in Manufacturing  
🔗 [GitHub](https://github.com/<your-username>) | 📧 <your.email@example.com>
