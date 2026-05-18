# ---------------- Streamlit App for Crash Analytics Project (Final Version) ----------------
# DharmpratapSingh Vaghela - AIT-582 Group 7
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
import seaborn as sns
import plotly.express as px
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
st.set_page_config(page_title="🚗 Crash Analytics App", layout="wide")

# ---------------- Session State for Navigation ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ---------------- Data Loading & Preprocessing ----------------
@st.cache_data
def load_preprocess_data():
    df = pd.read_csv("Fine-Tuned_Road_Accident_Dataset.csv")

    cat_cols = ['Road Condition', 'Weather Conditions', 'Urban/Rural',
                'Driver Age Group', 'Driver Gender', 'Vehicle Condition', 'Accident Severity']
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    features = ['Traffic Volume', 'Population Density', 'Speed Limit', 'Visibility Level',
                'Driver Alcohol Level', 'Driver Fatigue', 'Pedestrians Involved',
                'Number of Injuries', 'Number of Fatalities', 'Emergency Response Time',
                'Medical Cost', 'Economic Loss', 'Road Condition', 'Weather Conditions',
                'Urban/Rural', 'Driver Age Group', 'Driver Gender', 'Vehicle Condition']

    target = 'Accident Severity'

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

    return df, X_resampled, y_resampled, features

df, X_resampled, y_resampled, all_features = load_preprocess_data()

# ---------------- Cache Model Training ----------------
@st.cache_resource
def train_rq1_model(df):
    features = ['Driver Age Group', 'Driver Gender', 'Driver Alcohol Level', 'Driver Fatigue']
    X = df[features]
    y = df['Accident Severity']

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred, features

@st.cache_resource
def train_rq2_model(df):
    features = ['Traffic Volume', 'Population Density', 'Urban/Rural']
    X = df[features]
    y = df['Accident Severity']

    X = StandardScaler().fit_transform(X)
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, stratify=y_res, random_state=42)

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred, features

@st.cache_resource
def train_rq3_model(df):
    features = ['Weather Conditions', 'Road Condition', 'Visibility Level', 'Time of Day']

    df_copy = df.copy()
    le = LabelEncoder()
    df_copy['Time of Day'] = le.fit_transform(df_copy['Time of Day'])

    X = df_copy[features]
    y = df_copy['Accident Severity']

    X = StandardScaler().fit_transform(X)
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, stratify=y_res, random_state=42)

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred, features

@st.cache_resource
def train_rq4_model(X_resampled, y_resampled):
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42)

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X_test, y_test, y_pred

# ---------------- Display Metrics ----------------
def display_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    st.success(f"✅ Accuracy: {acc:.2%} | F1 Score (Macro): {f1:.2%}")

    with st.expander("🔍 View Detailed Classification Report"):
        st.text(classification_report(y_test, y_pred))

# ---------------- Research Question 1 ----------------
def run_rq1_logistic(df):
    st.markdown("### 📚 Research Question 1: Driver-Related Factors")
    st.markdown("""
    How do driver-related factors such as age, gender, alcohol consumption, and fatigue affect accident severity?

    ✅ **Model Used:** Logistic Regression (easy interpretability)

    ---
    """)

    model, X_test, y_test, y_pred, features = train_rq1_model(df)
    display_metrics(y_test, y_pred)

    st.markdown("### 📊 Driver Alcohol Level Distribution")
    st.markdown("This histogram shows how alcohol levels are distributed among drivers involved in accidents. Peaks at certain levels may highlight risky drinking patterns contributing to higher severity.")
    fig = px.histogram(df, x='Driver Alcohol Level', nbins=30, title="Driver Alcohol Level Histogram")
    st.plotly_chart(fig)

    st.markdown("### 📊 Driver Feature Importance (Logistic Regression Coefficients)")
    st.markdown("This chart shows which driver-related factors (age, gender, alcohol level, fatigue) have the highest positive or negative impact on accident severity.")
    coeff = model.coef_[0]
    importance_df = pd.DataFrame({"Feature": features, "Coefficient": coeff}).sort_values(by="Coefficient", ascending=False)
    fig2, ax2 = plt.subplots()
    ax2.barh(importance_df['Feature'], importance_df['Coefficient'])
    ax2.set_title("Driver Feature Importance (Logistic Regression)")
    st.pyplot(fig2)

# ---------------- Research Question 2 ----------------
def run_rq2_random_forest(df):
    st.markdown("### 📚 Research Question 2: Environmental Factors")
    st.markdown("""
    How do environmental factors like traffic volume, urban settings, and population density influence accident severity?

    ✅ **Model Used:** Random Forest (handles nonlinear relationships)

    ---
    """)

    model, X_test, y_test, y_pred, features = train_rq2_model(df)
    display_metrics(y_test, y_pred)

    st.markdown("### 📊 Traffic Volume by Accident Severity")
    st.markdown("This boxplot displays how traffic volume varies across accidents of different severity levels. Higher congestion may correlate with accident risk.")
    fig = px.box(df, x='Accident Severity', y='Traffic Volume', color='Accident Severity', title="Traffic Volume by Severity")
    st.plotly_chart(fig)

    st.markdown("### 📊 Environmental Feature Importance (Random Forest)")
    st.markdown("This bar chart ranks environmental factors such as traffic volume, urban/rural setting, and population density by their impact on accident severity.")
    importances = model.feature_importances_
    fig2, ax2 = plt.subplots()
    ax2.barh(features, importances)
    ax2.set_title("Environmental Feature Importance (Random Forest)")
    st.pyplot(fig2)

# ---------------- Research Question 3 ----------------
def run_rq3_random_forest(df):
    st.markdown("### 📚 Research Question 3: Temporal and Weather Impact")
    st.markdown("""
    How do weather conditions, road types, visibility, and time of day impact accident severity patterns?

    ✅ **Model Used:** Random Forest (captures feature interactions)

    ---
    """)

    model, X_test, y_test, y_pred, features = train_rq3_model(df)
    display_metrics(y_test, y_pred)

    st.markdown("### 📊 Time of Day vs Accident Severity")
    st.markdown("This boxplot illustrates how accident severity varies depending on the time of day — for example, nighttime crashes may be more severe.")
    df_copy = df.copy()
    le = LabelEncoder()
    df_copy['Time of Day'] = le.fit_transform(df_copy['Time of Day'])
    fig = px.box(df_copy, x='Accident Severity', y='Time of Day', color='Accident Severity', title="Time of Day by Severity")
    st.plotly_chart(fig)

    st.markdown("### 📊 Temporal and Weather Feature Importance (Random Forest)")
    st.markdown("This bar chart ranks temporal and weather factors by their contribution to accident severity based on the Random Forest model.")
    importances = model.feature_importances_
    fig2, ax2 = plt.subplots()
    ax2.barh(features, importances)
    ax2.set_title("Temporal and Weather Feature Importance (Random Forest)")
    st.pyplot(fig2)

# ---------------- Research Question 4 ----------------
def run_rq4_random_forest(X_resampled, y_resampled, all_features):
    st.markdown("### 📚 Research Question 4: Accident Severity Interaction")
    st.markdown("""
    What are the primary causes behind road accidents at different severity levels, and how do multiple factors interact?

    ✅ **Model Used:** Random Forest (best global performance)

    ---
    """)

    model, X_test, y_test, y_pred = train_rq4_model(X_resampled, y_resampled)
    display_metrics(y_test, y_pred)

    st.markdown("### 📊 Overall Feature Importance (Random Forest)")
    st.markdown("This feature importance plot shows which combined factors (drivers, environment, time, weather) most strongly predict accident severity.")
    importances = model.feature_importances_
    indices = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.array(all_features)[indices], importances[indices])
    ax.set_title("Overall Feature Importances - Random Forest")
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🔍 Model Interpretability: SHAP and PDP")

    col1, col2 = st.columns(2)

    with col1:
        st.image("example_shap_summary_plot.png", caption="SHAP Summary Plot - Explains model behavior", use_column_width=True)

    with col2:
        st.image("example_pdp_plot.png", caption="Partial Dependence Plot - Minor Accident Risk Factors", use_column_width=True)

# ---------------- Home Page Layout ----------------
# ---------------- Home Page Layout ----------------
if st.session_state.page == "Home":
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🚦 Crash Analytics Dashboard</h1>", unsafe_allow_html=True)

    # ✅ Corrected Project Introduction
    st.markdown("### 📄 About this Project")
    st.info("""
    Road traffic accidents remain a critical global issue, causing significant human suffering and economic losses.  
    Our project uses machine learning models to predict accident severity based on driver behavior, environmental factors, and temporal patterns.
    """)

    # ✅ Corrected Dataset Overview
    st.markdown("### 📂 Dataset Overview")
    st.success("""
    - **132,000** accident records  
    - **30** predictive features (Driver behavior, Environment, Time-based patterns)  
    - Data from **multiple countries** and **diverse road conditions**
    """)

    # Quick Metrics
    st.markdown("### 📈 Quick Dataset Statistics")
    total_records = df.shape[0]
    total_features = df.shape[1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{total_records}")
    col2.metric("Total Features", f"{total_features}")
    col3.metric("Missing Values", "0")

    st.markdown("---")
    st.markdown("### 📊 Top Insights from Dataset")

    # Top Causes
    top_causes = df['Accident Cause'].value_counts().head(3)
    st.markdown("**🔹 Top 3 Accident Causes:**")
    for i, (cause, count) in enumerate(top_causes.items(), start=1):
        st.markdown(f"**{i}.** {cause} ({count} cases)")

    # Top Driver Age Groups
    top_ages = df['Driver Age Group'].value_counts().head(3)
    st.markdown("**🔹 Top 3 Driver Age Groups Involved:**")
    for i, (age, count) in enumerate(top_ages.items(), start=1):
        st.markdown(f"**{i}.** {age} ({count} drivers)")

    # Accident Severity Pie Chart
    st.markdown("### 🚦 Accident Severity Distribution")
    severity_counts = df['Accident Severity'].value_counts()
    fig_pie = px.pie(names=severity_counts.index, values=severity_counts.values,
                     title="Distribution of Accident Severity",
                     color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_pie)

    st.markdown("""
    **Interpretation:**  
    This pie chart shows the proportion of accidents categorized into Minor, Moderate, and Severe severity levels.
    """)

    st.markdown("---")

    # Key Factors
    st.markdown("### 🧠 Key Factors Analyzed in This Study")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🚗 **Driver-Related:**")
        st.markdown("- Age\n- Gender\n- Alcohol Level\n- Fatigue")
    with col2:
        st.markdown("🌎 **Environmental:**")
        st.markdown("- Traffic Volume\n- Urban/Rural\n- Population Density")
    with col3:
        st.markdown("⏰ **Temporal & Weather:**")
        st.markdown("- Time of Day\n- Weather Conditions\n- Road Condition")

    # Start Button
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>🚀 Ready to explore the research questions?</h2>", unsafe_allow_html=True)

    if st.button("👉 Start Exploring Research Questions"):
        st.session_state.page = "ResearchQuestions"

# ---------------- Research Question Navigation ----------------
elif st.session_state.page == "ResearchQuestions":
    st.sidebar.markdown("<h2 style='color: #1ABC9C;'>📚 Research Questions</h2>", unsafe_allow_html=True)

    option = st.sidebar.radio("Choose Analysis:",
        ("🚗 RQ1: Driver-Related Factors",
         "🌎 RQ2: Environmental Factors",
         "⏰ RQ3: Temporal and Weather Impact",
         "⚡ RQ4: Accident Severity Interaction")
    )

    if option == "🚗 RQ1: Driver-Related Factors":
        st.markdown("<h2 style='color: #3498db;'>🔹 RQ1: Driver-Related Factors</h2>", unsafe_allow_html=True)
        run_rq1_logistic(df)

    elif option == "🌎 RQ2: Environmental Factors":
        st.markdown("<h2 style='color: #2ECC71;'>🔹 RQ2: Environmental Factors</h2>", unsafe_allow_html=True)
        run_rq2_random_forest(df)

    elif option == "⏰ RQ3: Temporal and Weather Impact":
        st.markdown("<h2 style='color: #F39C12;'>🔹 RQ3: Temporal and Weather Impact</h2>", unsafe_allow_html=True)
        run_rq3_random_forest(df)

    elif option == "⚡ RQ4: Accident Severity Interaction":
        st.markdown("<h2 style='color: #9B59B6;'>🔹 RQ4: Accident Severity Interaction</h2>", unsafe_allow_html=True)
        run_rq4_random_forest(X_resampled, y_resampled, all_features)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 14px;'>"
    "Developed️ by Team 7 | AIT582 | Spring 2025"
    "Aishwarya Sura, Adarsh Thunga , Cory Trainor, DharmpratapSingh ArunodaySingh Vaghela , Sai Sriram Uppada, Sampath Kalyan Vankayala , Surthesh Velu Samy , Vani Subadhra Yelleti " 
    "</div>",
    unsafe_allow_html=True
)