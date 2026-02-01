
# Credit Card Fraud Detection – Sampling Techniques Comparison

## 📌 Project Overview
Credit card fraud datasets are **highly imbalanced**, which negatively impacts machine learning model performance.  
This project focuses on **balancing the dataset using multiple sampling techniques** and **evaluating their effect on different ML models**.

The goal is to identify:
- Which **sampling technique** performs best
- Which **model** achieves the highest accuracy
- How sampling affects model stability and performance

---

## 📂 Dataset
- Source: Credit Card Fraud Dataset
- Target variable: `Class`
  - `0` → Non-Fraud
  - `1` → Fraud
- Original dataset is **highly imbalanced**

---

## 🛠️ Methodology

### 1️⃣ Data Preprocessing
- Loaded the dataset using Pandas
- Separated features (`X`) and target (`y`)
- Verified severe class imbalance

---

### 2️⃣ Sampling Techniques Applied
We applied **five different sampling techniques** to balance the dataset:

| Sampling Name | Technique |
|--------------|----------|
| Sampling1 | Random Over Sampling |
| Sampling2 | Random Under Sampling |
| Sampling3 | SMOTE |
| Sampling4 | SMOTE + Tomek Links |
| Sampling5 | SMOTE + ENN |

---

### 3️⃣ Machine Learning Models Used
Five ML models were trained on each sampled dataset:

| Model ID | Algorithm |
|--------|----------|
| M1 | Logistic Regression |
| M2 | Decision Tree |
| M3 | Random Forest |
| M4 | Support Vector Machine (SVM) |
| M5 | Naive Bayes |

---

### 4️⃣ Evaluation Metric
- **Accuracy (%)** was used to compare model performance
- Train-test split: **70% / 30%**

---

## 📊 Results

### 🔹 Accuracy Table
| Model | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|-----|----------|----------|----------|----------|----------|
| M1 | 91.70 | 66.67 | 90.39 | 92.17 | 96.24 |
| M2 | 98.47 | 66.67 | 98.25 | 98.66 | 98.55 |
| M3 | 99.78 | 50.00 | 99.34 | 99.55 | 99.42 |
| M4 | 68.56 | 16.67 | 67.69 | 66.67 | 69.65 |
| M5 | 78.17 | 50.00 | 84.50 | 89.26 | 87.86 |

---

### 🔹 Best Model per Sampling Technique
| Sampling Technique | Best Model |
|-------------------|-----------|
| Sampling1 | M3 |
| Sampling2 | M1 |
| Sampling3 | M3 |
| Sampling4 | M3 |
| Sampling5 | M3 |

---

## 📈 Visualizations

### Accuracy Comparison Across Models
![Accuracy Comparison](images/accuracy_comparison_models.png)

---

### Accuracy Heatmap
![Accuracy Heatmap](images/accuracy_heatmap.png)

---

### Sampling-wise Performance Visualization
![Sampling Comparison](images/sampling_comparison.png)

---

## 🏆 Key Outcomes & Observations

- **Random Forest (M3)** consistently achieved the **highest accuracy**
- **SMOTE-based techniques** (Sampling3, Sampling4, Sampling5) performed significantly better than Random Under Sampling
- **Random Under Sampling** caused major information loss and poor accuracy
- **SMOTE + Tomek Links (Sampling4)** produced the most stable results
- Best overall performance:
  - **Model:** Random Forest (M3)
  - **Sampling:** Sampling4 / Sampling5

---

## 📌 Conclusion
Balancing techniques play a **critical role** in fraud detection tasks.  
Advanced oversampling methods like **SMOTE variants** significantly improve performance, especially when paired with ensemble models like **Random Forest**.

---

## 🚀 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib

---

## 📎 How to Run
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
