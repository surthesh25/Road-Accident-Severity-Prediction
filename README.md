# 🚗 Road Accident Severity Analysis

A machine learning project that analyses road accident data to predict and understand **accident severity** using classification models and explainability tools (SHAP, LIME, PDP).

The project addresses four research questions across driver behaviour, environmental conditions, temporal/weather factors, and overall feature interactions.

---

## 📌 Research Questions

| # | Research Question | Model |
|---|---|---|
| **RQ1** | How do driver-related factors (age, gender, alcohol, fatigue) influence accident severity? | Logistic Regression |
| **RQ2** | How do environmental factors (traffic volume, population density, urban/rural) affect severity? | Random Forest |
| **RQ3** | What is the impact of temporal and weather conditions (weather, road condition, visibility, time of day) on severity? | Random Forest |
| **RQ4** | How do all factors interact to determine accident severity? | Random Forest + SHAP + PDP + LIME |

---

## 📁 Repository Structure

```
Crash-Analytics/
│
├── Dataset/                        # Raw data 
│   └── Road_Accident_Dataset.csv
│
├── src/
│   └── crash_analysis.py           # Main script: EDA → modelling → explainability
│
├── outputs/                        # Auto-generated plots on first run
│   ├── eda_severity_distribution.png
│   ├── eda_feature_distributions.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_alcohol_by_severity.png
│   ├── cm_rq1_logistic_regression.png
│   ├── rq1_feature_importance.png
│   ├── cm_rq2_random_forest.png
│   ├── rq2_feature_importance.png
│   ├── cm_rq3_random_forest.png
│   ├── rq3_feature_importance.png
│   ├── cm_rq4_overall_random_forest.png
│   ├── rq4_feature_importance.png
│   ├── rq4_shap_bar.png
│   ├── rq4_shap_beeswarm.png
│   ├── rq4_pdp.png
│   └── rq4_lime_explanation.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

**File:** `Road_Accident_Dataset.csv`

| Feature | Description |
|---|---|
| `Traffic Volume` | Number of vehicles on the road |
| `Population Density` | Population density of the area |
| `Speed Limit` | Posted speed limit at the accident location |
| `Visibility Level` | Visibility score at the time of accident |
| `Driver Alcohol Level` | Blood alcohol concentration of the driver |
| `Driver Fatigue` | Binary flag — driver fatigue reported |
| `Pedestrians Involved` | Number of pedestrians involved |
| `Number of Injuries` | Total injuries recorded |
| `Number of Fatalities` | Total fatalities recorded |
| `Emergency Response Time` | Time taken for emergency services to arrive |
| `Medical Cost` | Estimated medical cost of the accident |
| `Economic Loss` | Total economic loss from the accident |
| `Road Condition` | Road surface condition (categorical) |
| `Weather Conditions` | Weather at the time of accident (categorical) |
| `Urban/Rural` | Whether the accident occurred in an urban or rural area |
| `Driver Age Group` | Age group of the driver (categorical) |
| `Driver Gender` | Gender of the driver (categorical) |
| `Vehicle Condition` | Condition of the vehicle involved (categorical) |
| `Time of Day` | Time period of the accident (categorical) |
| `Accident Severity` | **Target** — severity class of the accident |

> **Note:** The dataset file is not included in this repository.
> Place `Road_Accident_Dataset.csv` inside the `Dataset/` folder before running the script.

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Crash-Analytics.git
cd Crash-Analytics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place your dataset file in the `Dataset/` folder:

```
Dataset/
└── Road_Accident_Dataset.csv
```

### 4. Run the analysis

```bash
python src/crash_analysis.py
```

The `outputs/` folder is created automatically with all plots and figures.

---

## 🔍 Analysis Overview

### 🧹 Data Quality & EDA
- Missing value check and outlier detection (IQR method)
- Class distribution of accident severity
- Feature distributions (histograms with KDE)
- Correlation heatmap across all features
- Driver alcohol level vs severity (box plot)

### 🤖 Modelling Pipeline (per RQ)
Each research question follows the same pipeline:

```
Feature Selection → Scaling (StandardScaler) → SMOTE (class balancing)
→ Train/Test Split (70/30) → Model Training → Evaluation
```

**Metrics reported for every model:**
- Accuracy
- F1 Score (Macro)
- ROC-AUC (One-vs-Rest)
- 5-Fold Stratified Cross-Validation Accuracy
- Confusion Matrix (heatmap)
- Feature Importance chart

### 🔬 Explainability (RQ4 only)
| Tool | Output |
|---|---|
| **SHAP** | Bar chart (mean \|SHAP\|) + Beeswarm plot |
| **PDP** | Partial dependence plots for top features |
| **LIME** | Local explanation for an individual prediction |

---

## 📈 Model Summary

| RQ | Model | Features Used |
|---|---|---|
| RQ1 | Logistic Regression | Driver Age Group, Driver Gender, Driver Alcohol Level, Driver Fatigue |
| RQ2 | Random Forest | Traffic Volume, Population Density, Urban/Rural |
| RQ3 | Random Forest | Weather Conditions, Road Condition, Visibility Level, Time of Day |
| RQ4 | Random Forest | All 18 features |

---

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| [pandas](https://pandas.pydata.org/) | Data loading and manipulation |
| [NumPy](https://numpy.org/) | Numerical computation |
| [scikit-learn](https://scikit-learn.org/) | ML models, metrics, preprocessing |
| [imbalanced-learn](https://imbalanced-learn.org/) | SMOTE for class imbalance |
| [SHAP](https://shap.readthedocs.io/) | Global and local model explainability |
| [LIME](https://lime-ml.readthedocs.io/) | Local instance-level explanation |
| [Matplotlib](https://matplotlib.org/) | Plotting |
| [Seaborn](https://seaborn.pydata.org/) | Statistical visualisations |

---

## 📄 License

This project is for academic and research purposes.
Code is available under the [MIT License](LICENSE).
