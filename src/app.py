# =============================================================================
# Road Accident Severity — Streamlit Dashboard
# =============================================================================
# Interactive web app for exploring road accident severity predictions
# across four research questions using ML models and explainability tools.
#
# Run with:
#   streamlit run src/app.py
#
# Research Questions:
#   RQ1: Driver-related factors         → Logistic Regression
#   RQ2: Environmental factors          → Random Forest
#   RQ3: Temporal & weather factors     → Random Forest
#   RQ4: Overall factor interaction     → Random Forest + SHAP + PDP
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────

import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "🚗 Crash Analytics Dashboard",
    page_icon  = "🚦",
    layout     = "wide"
)

# ── Session State for Navigation ──────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "Home"


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

@st.cache_data
def load_preprocess_data():
    """
    Load and preprocess the road accident dataset.

    Encodes categorical columns, scales features, and applies SMOTE
    for class balancing.

    Returns
    -------
    df           : DataFrame  – Encoded dataset
    X_resampled  : np.ndarray – SMOTE-resampled feature matrix
    y_resampled  : np.ndarray – SMOTE-resampled target labels
    features     : list       – Feature column names
    """
    df = pd.read_csv("Dataset/Fine-Tuned_Road_Accident_Dataset.csv")

    cat_cols = [
        'Road Condition', 'Weather Conditions', 'Urban/Rural',
        'Driver Age Group', 'Driver Gender', 'Vehicle Condition', 'Accident Severity'
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    # Encode Time of Day if present
    if 'Time of Day' in df.columns and df['Time of Day'].dtype == object:
        df['Time of Day'] = LabelEncoder().fit_transform(df['Time of Day'])

    features = [
        'Traffic Volume', 'Population Density', 'Speed Limit', 'Visibility Level',
        'Driver Alcohol Level', 'Driver Fatigue', 'Pedestrians Involved',
        'Number of Injuries', 'Number of Fatalities', 'Emergency Response Time',
        'Medical Cost', 'Economic Loss', 'Road Condition', 'Weather Conditions',
        'Urban/Rural', 'Driver Age Group', 'Driver Gender', 'Vehicle Condition'
    ]
    features = [f for f in features if f in df.columns]

    X        = df[features]
    y        = df['Accident Severity']
    X_scaled = StandardScaler().fit_transform(X)

    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

    return df, X_resampled, y_resampled, features


df, X_resampled, y_resampled, all_features = load_preprocess_data()


# =============================================================================
# CACHED MODEL TRAINING
# =============================================================================

@st.cache_resource
def train_rq1_model(df):
    features = [f for f in ['Driver Age Group', 'Driver Gender',
                             'Driver Alcohol Level', 'Driver Fatigue'] if f in df.columns]
    X = StandardScaler().fit_transform(df[features])
    y = df['Accident Severity']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, model.predict(X_test), features


@st.cache_resource
def train_rq2_model(df):
    features = [f for f in ['Traffic Volume', 'Population Density', 'Urban/Rural']
                if f in df.columns]
    X     = StandardScaler().fit_transform(df[features])
    y     = df['Accident Severity']
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, stratify=y_res, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, model.predict(X_test), features


@st.cache_resource
def train_rq3_model(df):
    features = [f for f in ['Weather Conditions', 'Road Condition',
                             'Visibility Level', 'Time of Day'] if f in df.columns]
    X     = StandardScaler().fit_transform(df[features])
    y     = df['Accident Severity']
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, stratify=y_res, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, model.predict(X_test), features


@st.cache_resource
def train_rq4_model(X_resampled, y_resampled):
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, model.predict(X_test)


# =============================================================================
# HELPER — METRICS DISPLAY
# =============================================================================

def display_metrics(y_test, y_pred, model_name="Model"):
    """
    Display accuracy, F1 score, confusion matrix, and classification report
    in the Streamlit UI.
    """
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')

    col1, col2 = st.columns(2)
    col1.metric("✅ Accuracy",        f"{acc:.2%}")
    col2.metric("📊 F1 Score (Macro)", f"{f1:.2%}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5, ax=ax)
    ax.set_title(f'Confusion Matrix — {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    plt.close()

    with st.expander("🔍 View Detailed Classification Report"):
        st.text(classification_report(y_test, y_pred))


# =============================================================================
# RESEARCH QUESTION PAGES
# =============================================================================

def run_rq1(df):
    st.markdown("<h2 style='color:#3498db;'>🚗 RQ1 — Driver-Related Factors</h2>",
                unsafe_allow_html=True)
    st.markdown("""
    **Research Question:** How do driver-related factors such as age, gender,
    alcohol consumption, and fatigue affect accident severity?

    **Model:** Logistic Regression — chosen for its interpretability via coefficients.
    """)
    st.divider()

    model, X_test, y_test, y_pred, features = train_rq1_model(df)
    display_metrics(y_test, y_pred, "RQ1 — Logistic Regression")

    st.markdown("#### 📊 Driver Alcohol Level Distribution")
    st.caption("Peaks at higher alcohol levels indicate risky drinking patterns linked to greater severity.")
    fig = px.histogram(df, x='Driver Alcohol Level', nbins=30,
                       title="Driver Alcohol Level Distribution",
                       color_discrete_sequence=['#3498db'])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📊 Driver Feature Importance (Logistic Regression Coefficients)")
    st.caption("Positive coefficients increase predicted severity; negative ones reduce it.")
    coeff_df = pd.DataFrame({
        'Feature'    : features,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient')

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in coeff_df['Coefficient']]
    ax2.barh(coeff_df['Feature'], coeff_df['Coefficient'], color=colors, edgecolor='white')
    ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_title("Driver Feature Importance (Logistic Regression)")
    ax2.set_xlabel("Coefficient")
    st.pyplot(fig2)
    plt.close()


def run_rq2(df):
    st.markdown("<h2 style='color:#2ecc71;'>🌎 RQ2 — Environmental Factors</h2>",
                unsafe_allow_html=True)
    st.markdown("""
    **Research Question:** How do environmental factors like traffic volume,
    urban settings, and population density influence accident severity?

    **Model:** Random Forest — handles nonlinear relationships between features.
    """)
    st.divider()

    model, X_test, y_test, y_pred, features = train_rq2_model(df)
    display_metrics(y_test, y_pred, "RQ2 — Random Forest")

    st.markdown("#### 📊 Traffic Volume by Accident Severity")
    st.caption("Higher traffic congestion may correlate with increased accident risk and severity.")
    fig = px.box(df, x='Accident Severity', y='Traffic Volume',
                 color='Accident Severity',
                 title="Traffic Volume by Accident Severity",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📊 Environmental Feature Importance (Random Forest)")
    imp_df = pd.DataFrame({
        'Feature'   : features,
        'Importance': model.feature_importances_
    }).sort_values('Importance')

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.barh(imp_df['Feature'], imp_df['Importance'], color='#2ecc71', edgecolor='white')
    ax2.set_title("Environmental Feature Importance")
    ax2.set_xlabel("Importance Score")
    st.pyplot(fig2)
    plt.close()


def run_rq3(df):
    st.markdown("<h2 style='color:#f39c12;'>⏰ RQ3 — Temporal & Weather Impact</h2>",
                unsafe_allow_html=True)
    st.markdown("""
    **Research Question:** How do weather conditions, road types, visibility,
    and time of day impact accident severity patterns?

    **Model:** Random Forest — captures complex feature interactions.
    """)
    st.divider()

    model, X_test, y_test, y_pred, features = train_rq3_model(df)
    display_metrics(y_test, y_pred, "RQ3 — Random Forest")

    st.markdown("#### 📊 Time of Day vs Accident Severity")
    st.caption("Nighttime and early morning hours may show higher severity levels.")
    fig = px.box(df, x='Accident Severity', y='Time of Day',
                 color='Accident Severity',
                 title="Time of Day by Accident Severity",
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📊 Weather & Temporal Feature Importance (Random Forest)")
    imp_df = pd.DataFrame({
        'Feature'   : features,
        'Importance': model.feature_importances_
    }).sort_values('Importance')

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.barh(imp_df['Feature'], imp_df['Importance'], color='#f39c12', edgecolor='white')
    ax2.set_title("Temporal & Weather Feature Importance")
    ax2.set_xlabel("Importance Score")
    st.pyplot(fig2)
    plt.close()


def run_rq4(X_resampled, y_resampled, all_features):
    st.markdown("<h2 style='color:#9b59b6;'>⚡ RQ4 — Overall Factor Interaction</h2>",
                unsafe_allow_html=True)
    st.markdown("""
    **Research Question:** What are the primary causes behind road accidents at
    different severity levels, and how do multiple factors interact?

    **Model:** Random Forest on all 18 features + SHAP + PDP explainability.
    """)
    st.divider()

    model, X_test, y_test, y_pred = train_rq4_model(X_resampled, y_resampled)
    display_metrics(y_test, y_pred, "RQ4 — Random Forest (All Features)")

    # --- Overall feature importance ---
    st.markdown("#### 📊 Overall Feature Importance (Random Forest)")
    st.caption("Ranks all 18 features by their contribution to predicting accident severity.")
    importances = model.feature_importances_
    indices     = np.argsort(importances)
    colors      = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(indices)))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(np.array(all_features)[indices], importances[indices],
            color=colors, edgecolor='white')
    ax.set_title("Overall Feature Importances — Random Forest")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)
    plt.close()

    st.divider()

    # --- SHAP ---
    st.markdown("#### 🔍 SHAP Explainability")
    st.caption("SHAP values show how each feature pushes the model's prediction higher or lower.")

    with st.spinner("Computing SHAP values (this may take a moment)..."):
        explainer   = shap.TreeExplainer(model)
        shap_sample = X_test[:300]
        shap_values = explainer.shap_values(shap_sample)

        fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, shap_sample,
                          feature_names=all_features, show=False)
        plt.title("SHAP Beeswarm — Feature Impact on Severity")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    # --- PDP ---
    st.markdown("#### 📉 Partial Dependence Plots")
    st.caption("Shows the marginal effect of each feature on the predicted severity.")

    pdp_features = [f for f in ['Driver Alcohol Level', 'Visibility Level', 'Speed Limit']
                    if f in all_features]
    pdp_indices  = [all_features.index(f) for f in pdp_features]

    if pdp_indices:
        fig_pdp, ax_pdp = plt.subplots(figsize=(12, 4))
        PartialDependenceDisplay.from_estimator(
            model, X_test[:300], pdp_indices,
            feature_names=all_features, grid_resolution=20, ax=ax_pdp
        )
        plt.suptitle("Partial Dependence Plots — Key Features", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig_pdp)
        plt.close()


# =============================================================================
# HOME PAGE
# =============================================================================

if st.session_state.page == "Home":
    st.markdown(
        "<h1 style='text-align:center; color:#2E86C1;'>🚦 Crash Analytics Dashboard</h1>",
        unsafe_allow_html=True
    )

    st.markdown("### 📄 About this Project")
    st.info("""
    Road traffic accidents remain a critical global issue, causing significant human suffering
    and economic losses. This dashboard uses machine learning to predict accident severity
    based on driver behaviour, environmental conditions, and temporal patterns.
    """)

    st.markdown("### 📂 Dataset Overview")
    st.success("""
    - **132,000** accident records
    - **30** predictive features (driver behaviour, environment, time-based patterns)
    - Data from **multiple countries** and **diverse road conditions**
    """)

    # --- Quick metrics ---
    st.markdown("### 📈 Dataset Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records",  f"{df.shape[0]:,}")
    col2.metric("Total Features", f"{df.shape[1]}")
    col3.metric("Missing Values", "0")

    st.divider()

    # --- Top insights ---
    st.markdown("### 📊 Quick Insights")
    col_a, col_b = st.columns(2)

    with col_a:
        if 'Accident Cause' in df.columns:
            st.markdown("**🔹 Top 3 Accident Causes:**")
            for i, (cause, count) in enumerate(
                df['Accident Cause'].value_counts().head(3).items(), 1
            ):
                st.markdown(f"**{i}.** {cause} — {count:,} cases")

    with col_b:
        st.markdown("**🔹 Top 3 Driver Age Groups Involved:**")
        for i, (age, count) in enumerate(
            df['Driver Age Group'].value_counts().head(3).items(), 1
        ):
            st.markdown(f"**{i}.** Age Group {age} — {count:,} drivers")

    # --- Severity pie chart ---
    st.markdown("### 🚦 Accident Severity Distribution")
    severity_counts = df['Accident Severity'].value_counts()
    fig_pie = px.pie(
        names  = severity_counts.index,
        values = severity_counts.values,
        title  = "Distribution of Accident Severity",
        color_discrete_sequence = px.colors.sequential.RdBu
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    st.caption("Proportion of accidents categorised into Minor, Moderate, and Severe levels.")

    st.divider()

    # --- Key factors summary ---
    st.markdown("### 🧠 Key Factors Analysed")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🚗 **Driver-Related:**")
        st.markdown("- Age Group\n- Gender\n- Alcohol Level\n- Fatigue")
    with col2:
        st.markdown("🌎 **Environmental:**")
        st.markdown("- Traffic Volume\n- Urban/Rural\n- Population Density")
    with col3:
        st.markdown("⏰ **Temporal & Weather:**")
        st.markdown("- Time of Day\n- Weather Conditions\n- Road Condition\n- Visibility")

    st.divider()
    st.markdown(
        "<h3 style='text-align:center;'>🚀 Ready to explore the research questions?</h3>",
        unsafe_allow_html=True
    )
    if st.button("👉 Start Exploring", use_container_width=True):
        st.session_state.page = "ResearchQuestions"
        st.rerun()


# =============================================================================
# RESEARCH QUESTION NAVIGATION
# =============================================================================

elif st.session_state.page == "ResearchQuestions":

    st.sidebar.markdown(
        "<h2 style='color:#1ABC9C;'>📚 Research Questions</h2>",
        unsafe_allow_html=True
    )

    if st.sidebar.button("🏠 Back to Home"):
        st.session_state.page = "Home"
        st.rerun()

    option = st.sidebar.radio(
        "Choose Analysis:",
        (
            "🚗 RQ1: Driver-Related Factors",
            "🌎 RQ2: Environmental Factors",
            "⏰ RQ3: Temporal & Weather Impact",
            "⚡ RQ4: Overall Factor Interaction"
        )
    )

    if   option == "🚗 RQ1: Driver-Related Factors":
        run_rq1(df)
    elif option == "🌎 RQ2: Environmental Factors":
        run_rq2(df)
    elif option == "⏰ RQ3: Temporal & Weather Impact":
        run_rq3(df)
    elif option == "⚡ RQ4: Overall Factor Interaction":
        run_rq4(X_resampled, y_resampled, all_features)


# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:13px;'>
        Developed by <b>Team 7</b> | AIT-582 | Spring 2025<br>
        Aishwarya Sura · Adarsh Thunga · Cory Trainor · DharmpratapSingh Vaghela ·
        Sai Sriram Uppada · Sampath Kalyan Vankayala · Surthesh Velu Samy · Vani Subadhra Yelleti
    </div>
    """,
    unsafe_allow_html=True
)
