# Industrial AI Projects – Chamber Pressure Stability & Tool Wear Prediction

This repository showcases two applied machine learning projects focused on predictive analytics for semiconductor and CNC manufacturing systems.

Both projects demonstrate how domain-driven feature engineering and interpretable ML improve equipment reliability, yield, and process consistency.

---

## 🧠 Project 1 – Chamber Pressure Stability

### Overview
Predicts whether a semiconductor process chamber will remain pressure-stable during etch or deposition.  
Even minor drifts (±2 mTorr) can cause wafer defects or yield loss.  
This model forecasts instability from process variables before it happens.

### Approach
1. **Data Cleaning** – Median imputation and outlier clipping (IQR)  
2. **Feature Engineering** –  
   - Gas ratios (`O₂/CF₄`, `Ar/CF₄`)  
   - Pump efficiency and valve normalization  
   - Flow balance and instability index  
3. **Modeling** – Linear → Random Forest → XGBoost → Gradient Boosting  
4. **Evaluation** – RMSE, MAE, and R² metrics  
5. **Interpretation** – SHAP analysis confirms domain logic:  
   high `O₂/CF₄` ratios and closed valves often destabilize pressure.

### Deliverables
| File | Description |
|------|--------------|
| `chamberpressurestability.py` | End-to-end training and evaluation script |
| `notebooks/` | Colab or Jupyter notebooks with data exploration and plots |
| `data/` | Example CSV logs of chamber parameters |

---

## ⚙️ Project 2 – Tool Wear Prediction

### Overview
Predicts **CNC tool wear state (Usable = 0 / Worn = 1)** using machining sensor data such as spindle load, vibration, and temperature.  
Goal: identify worn tools early to reduce downtime and scrap.

### Feature Engineering
| Feature | Description |
|----------|-------------|
| `hours_cumulative` | Total tool usage hours |
| `cycle_index` | Operation cycle count |
| `temp_slope` | Rate of temperature rise per cycle |
| `load_variation` | Rolling std of spindle load |
| `vibration_variation` | Rolling std of vibration |
| `stress_factor` | `spindle_load × hours_run` |
| `thermo_mech_index` | `spindle_load × temperature` |

### Models Benchmarked
| Model | Type |
|--------|------|
| Logistic Regression | Linear baseline |
| Decision Tree | Simple non-linear model |
| Random Forest | Bagging ensemble |
| Gradient Boosting | Boosted ensemble |
| XGBoost | Regularized boosted trees |
| CatBoost | Gradient boosting optimized for noisy data |

### Evaluation (5-Fold Stratified CV)
| Model | AUC | Recall (Worn) | F1 (Worn) | Accuracy |
|--------|-----|----------------|------------|-----------|
| **XGBoost** | 0.74 | 0.67 | **0.66** | 0.72 |
| CatBoost | 0.73 | **0.68** | **0.66** | 0.71 |
| Gradient Boosting | 0.72 | 0.64 | 0.64 | 0.70 |
| Random Forest | 0.69 | 0.61 | 0.60 | 0.69 |
| Logistic Regression | 0.61 | 0.48 | 0.50 | 0.67 |
| Decision Tree | 0.58 | 0.46 | 0.48 | 0.66 |

### Deliverables
| File | Description |
|------|--------------|
| `tool_wear_prediction.py` | Full multi-model comparison pipeline |
| `notebooks/` | Colab or Jupyter notebooks with all plots and outputs |
| `data/` | Example machining sensor dataset |

---

## 🧰 Common Tech Stack

| Category | Tools |
|-----------|--------|
| Language | Python 3.10 |
| Libraries | pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost, catboost, shap |
| Environment | Google Colab / Jupyter |
| Version Control | GitHub |

---

## 🔍 Key Insights

- **Derived features** (ratios, rolling variations, composite indices) outperform raw sensor data.  
- **Tree-based ensembles** like Gradient Boosting and XGBoost consistently achieved the best balance of recall and interpretability.  
- **Explainable AI (SHAP)** validated that the models’ top features align with physical intuition — gas ratios and valve position for pressure, vibration and thermal indices for wear.

---


### 👨‍💻 Author
**<YOUR NAME>**  
Machine Learning Engineer | Applied AI in Manufacturing & Semiconductor Systems  
📧 <your.email@example.com> • 🔗 [GitHub](https://github.com/<your-username>)
