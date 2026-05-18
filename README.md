# 🚗 Road Accident Severity Analysis

A machine learning project that analyses road accident data to **predict and explain accident severity** using classification models and explainability tools (SHAP, LIME, PDP).

The project includes two components:
- **`crash_analysis.py`** — Full offline analysis script with EDA, modelling, and evaluation
- **`app.py`** — Interactive Streamlit dashboard for exploring results visually

---

## 📌 Research Questions

| # | Research Question | Model |
|---|---|---|
| **RQ1** | How do driver-related factors (age, gender, alcohol, fatigue) influence accident severity? | Logistic Regression |
| **RQ2** | How do environmental factors (traffic volume, population density, urban/rural) affect severity? | Random Forest |
| **RQ3** | What is the impact of temporal and weather conditions on severity? | Random Forest |
| **RQ4** | How do all factors interact to determine accident severity? | Random Forest + SHAP + PDP + LIME |

---

## 📁 Repository Structure

```
Crash-Analytics/
│
├── Dataset/
│   ├── Road_Accident_Dataset.csv            # Original dataset
│   └── Fine-Tuned_Road_Accident_Dataset.csv # Fine-tuned dataset (used by app)
│
├── src/
│   ├── crash_analysis.py                    # Offline analysis: EDA → modelling → explainability
│   └── app.py                               # Streamlit interactive dashboard
│
├── outputs/                                 # Auto-generated plots (crash_analysis.py)
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

**File:** `Road_Accident_Dataset.csv` / `Fine-Tuned_Road_Accident_Dataset.csv`

| Feature | Description |
|---|---|
| `Traffic Volume` | Number of vehicles on the road |
| `Population Density` | Population density of the area |
| `Speed Limit` | Posted speed limit at the accident location |
| `Visibility Level` | Visibility score at time of accident |
| `Driver Alcohol Level` | Blood alcohol concentration of the driver |
| `Driver Fatigue` | Binary flag — driver fatigue reported |
| `Pedestrians Involved` | Number of pedestrians involved |
| `Number of Injuries` | Total injuries recorded |
| `Number of Fatalities` | Total fatalities recorded |
| `Emergency Response Time` | Time for emergency services to arrive |
| `Medical Cost` | Estimated medical cost of the accident |
| `Economic Loss` | Total economic loss from the accident |
| `Road Condition` | Road surface condition (categorical) |
| `Weather Conditions` | Weather at time of accident (categorical) |
| `Urban/Rural` | Urban or rural accident location |
| `Driver Age Group` | Age group of the driver (categorical) |
| `Driver Gender` | Gender of the driver (categorical) |
| `Vehicle Condition` | Condition of the vehicle (categorical) |
| `Time of Day` | Time period of the accident (categorical) |
| `Accident Severity` | **Target** — severity class of the accident |

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

### 3. Run the offline analysis script

```bash
python src/crash_analysis.py
```

Outputs are saved automatically to the `outputs/` folder.

### 4. Run the Streamlit dashboard

```bash
streamlit run src/app.py
```

Opens at `http://localhost:8501` in your browser.

---

## 🖥️ Streamlit Dashboard

The interactive dashboard has two sections:

**🏠 Home Page**
- Project overview and dataset summary
- Quick statistics (records, features, missing values)
- Top accident causes and driver age groups
- Accident severity pie chart
- Key factors summary

**📚 Research Questions (Sidebar Navigation)**

| Page | Content |
|---|---|
| RQ1 — Driver Factors | Metrics, alcohol distribution, logistic regression coefficients |
| RQ2 — Environmental | Metrics, traffic volume boxplot, RF feature importance |
| RQ3 — Weather/Time | Metrics, time of day boxplot, RF feature importance |
| RQ4 — All Factors | Metrics, overall importance, SHAP beeswarm, PDP plots |

---

## 🔍 Analysis Overview (`crash_analysis.py`)

### Modelling Pipeline (per RQ)
```
Feature Selection → StandardScaler → SMOTE → Train/Test Split (70/30)
→ Model Training → Evaluation (Accuracy, F1, ROC-AUC, 5-Fold CV)
→ Confusion Matrix → Feature Importance
```

### Explainability Tools (RQ4)

| Tool | Description |
|---|---|
| **SHAP** | Global feature impact via beeswarm + bar chart |
| **PDP** | Marginal effect of top features on severity |
| **LIME** | Local explanation for individual predictions |

---

## 📈 Model Summary

| RQ | Model | Features |
|---|---|---|
| RQ1 | Logistic Regression | Driver Age Group, Gender, Alcohol Level, Fatigue |
| RQ2 | Random Forest | Traffic Volume, Population Density, Urban/Rural |
| RQ3 | Random Forest | Weather Conditions, Road Condition, Visibility, Time of Day |
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
| [Streamlit](https://streamlit.io/) | Interactive web dashboard |
| [Plotly](https://plotly.com/python/) | Interactive charts in dashboard |
| [Matplotlib](https://matplotlib.org/) | Static visualisations |
| [Seaborn](https://seaborn.pydata.org/) | Statistical visualisations |

---

## 👥 Team

Developed by **Team 7** | AIT-582 | Spring 2025

Aishwarya Sura · Adarsh Thunga · Cory Trainor · DharmpratapSingh Vaghela ·
Sai Sriram Uppada · Sampath Kalyan Vankayala · Surthesh Velu Samy · Vani Subadhra Yelleti

---

## 📄 License

This project is for academic and research purposes.
Code is available under the [MIT License](LICENSE).
