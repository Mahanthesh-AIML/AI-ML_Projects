# 🔍 Machine Sensor Data Clustering (K-Means & DBSCAN)

This project applies **unsupervised learning** techniques to cluster machine operating states using real or synthetic sensor data.  
The goal is to uncover hidden structure in the data — identifying **normal**, **transient**, and **anomalous** behavior patterns without labeled outcomes.

---

## 🎯 Objective

Industrial systems generate massive amounts of sensor data — temperature, vibration, pressure, torque, and flow.  
Here, we use **K-Means** and **DBSCAN** clustering to:
- Group similar machine operating states.
- Detect abnormal patterns indicating instability or wear.
- Visualize high-dimensional relationships through PCA.

---

## 🧠 Techniques Used

| Technique | Purpose |
|------------|----------|
| **K-Means Clustering** | Partition data into *k* distinct groups using Euclidean distance. |
| **DBSCAN** | Density-based clustering to detect outliers and arbitrary-shaped clusters. |
| **PCA (Principal Component Analysis)** | Dimensionality reduction for visualization. |
| **Silhouette Score** | Quantify the quality of cluster separation. |
| **StandardScaler** | Normalize features before clustering. |

---

## 📊 Workflow

1. **Data Loading & Cleaning**
   - Handle missing values.
   - Normalize all numeric features.

2. **Exploratory Data Analysis (EDA)**
   - Pair plots, histograms, and correlation checks.
   - PCA scatter plots to understand variance.

3. **K-Means Clustering**
   - Use the **Elbow method** and **Silhouette score** to find optimal `k`.
   - Assign cluster labels and visualize with PCA.

4. **DBSCAN Clustering**
   - Tune `eps` and `min_samples`.
   - Identify high-density clusters and noise points.

5. **Evaluation**
   - Compare cluster metrics across algorithms.
   - Visualize cluster centers and boundaries.

6. **Interpretation**
   - Examine average feature values per cluster.
   - Derive business meaning (e.g., idle vs. high-load operation).

---

## 🧩 Example Results

| Algorithm | Silhouette Score | # Clusters | Outliers Detected |
|:-----------|:----------------:|:-----------:|:-----------------:|
| K-Means | 0.62 | 3 | 0 |
| DBSCAN | 0.58 | 4 | 112 |

> These results demonstrate clear separability between normal and high-stress operating modes, with DBSCAN detecting several low-density outliers corresponding to abnormal machine cycles.

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|-------|
| Language | Python 3.10 |
| ML Libraries | Scikit-Learn, NumPy, Pandas |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dimensionality Reduction | PCA |
| Environment | Google Colab / Jupyter |



