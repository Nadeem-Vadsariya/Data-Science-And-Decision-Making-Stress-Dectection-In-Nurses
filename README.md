# 🏥 Continuous Stress Detection in Hospital Nurses — Machine Learning Implementation


## 📌 Project Overview
Hospital nurses face high-pressure environments where acute and chronic stress can affect clinical decision-making, patient care, and personal well-being. 

This project implements an end-to-end machine learning classification pipeline using a **multimodal wearable physiological sensor dataset** collected from **15 hospital nurses** alongside clinical self-reported survey data (`SurveyResults.xlsx`). By aggregating over **514,000+ raw sensor records** into clean subject-level physiological profiles, the pipeline predicts nurse stress levels (**Low**, **Moderate**, **High**) to support proactive healthcare workforce management.

### 🔗 Related Repository
* **Part 1 — Exploratory Data Analysis (EDA):** [`Data-Science-And-Decision-Making-Stress-Dectection-In-Nurses-EDA`]([Insert-Link-To-Your-EDA-Repo])

---

## 📊 Dataset Architecture & Signals
The raw dataset (`Stress_dataset`) consists of multi-rate physiological signal recordings across **15 unique nurse IDs** (`15`, `5C`, `6B`, `6D`, `7A`, `7E`, `83`, `8B`, `94`, `BG`, `CE`, `DF`, `E4`, `EG`, `F5`), paired with survey stress logs:

* **Physiological Sensor Streams (514,675 rows):**
  * **ACC (`X, Y, Z`):** 3-axis accelerometer movement data
  * **BVP:** Blood Volume Pulse
  * **EDA:** Electrodermal Activity (Galvanic Skin Response)
  * **HR:** Heart Rate (bpm)
  * **TEMP:** Peripheral Skin Temperature (°C)
* **Target Labels (`SurveyResults.xlsx`):** Self-reported stress levels mapped into three discrete classes: `0.0 → Low`, `1.0 → Moderate`, and `2.0 → High`.

---

## 🛠️ Data Engineering & ML Pipeline

### 1. Multi-Signal Ingestion & Synchronization
* Automated directory traversal across all 15 nurse folders to parse individual signal files (`ACC.csv`, `BVP.csv`, `EDA.csv`, `HR.csv`, `TEMP.csv`).
* Extracted initial Unix timestamps and sampling rates (`start_timestamp + i / sample_rate`) to reconstruct time-indexed DataFrames before consolidating into a unified master dataset (`mergedcsv.csv`).

### 2. Signal Compression & Missing Value Imputation
* **Median Signal Aggregation:** Compressed 514,675 high-frequency sensor rows into robust subject-level median profiles (`median_X`, `median_Y`, `median_Z`, `median_BVP`, `median_EDA`, `median_HR`, `median_TEMP`) per nurse ID to eliminate noise and computational overhead.
* **Linear Interpolation:** Handled missing Heart Rate (`HR`) records across subjects (`E4`, `EG`, `F5`) using linear interpolation (`interpolate(method='linear')`).
* **Target Alignment:** Computed the modal stress level per nurse from `SurveyResults.xlsx` and performed an inner merge on nurse `id`.

### 3. Correlation-Based Feature Selection
* Computed a Pearson correlation matrix across all aggregated biometric features.
* Selected the three strongest physiologically correlated predictors—**`median_EDA`**, **`median_HR`**, and **`median_TEMP`** (showing strong positive inter-correlations up to `0.67`)—aligning with established biomedical stress research.

### 4. Class Balancing & Preprocessing
* **Oversampling:** Applied `RandomOverSampler` (`imblearn`) to balance class representation across `High`, `Moderate`, and `Low` stress categories.
* **Train-Test Split & Scaling:** Partitioned data (`80%` train / `20%` test) and standardized features using `StandardScaler`.

---

## 📈 Model Evaluation & Results

Two supervised classification algorithms were trained and evaluated using **5-Fold Cross-Validation**, test set accuracy, confusion matrices, and precision/recall/F1 metrics:

| Model | 5-Fold CV Mean Score | Test Accuracy | Macro Avg F1-Score | Selection Status |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **80.0%** (`0.80`) | **83.3%** (`0.833`) | **0.82** | ✅ **Selected Model** |
| **K-Nearest Neighbors (KNN)** | 75.0% (`0.75`) | 66.7% (`0.667`) | 0.53 | ❌ Baseline Comparison |

### Key Takeaways
* **Winning Model:** **Logistic Regression** outperformed KNN across both 5-fold cross-validation (`0.80` vs. `0.75`) and holdout test accuracy (`83.3%` vs. `66.7%`), achieving `1.00` precision on `High` stress and `1.00` F1-score on `Moderate` stress.
* **Biometric Insights:** Elevated **Electrodermal Activity (`median_EDA`)**, **Heart Rate (`median_HR`)**, and **Skin Temperature (`median_TEMP`)** served as reliable physiological markers for distinguishing occupational stress states.

---

## 💻 Tech Stack
* **Language:** Python 3.9
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning & Preprocessing:** Scikit-learn (`LogisticRegression`, `KNeighborsClassifier`, `StandardScaler`, `train_test_split`, `cross_val_score`), Imbalanced-learn (`RandomOverSampler`)
* **Environment:** Jupyter Notebook
