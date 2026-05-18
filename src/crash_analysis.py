# =============================================================================
# Road Accident Severity Analysis — Python
# =============================================================================
# This script analyses road accident data to predict and understand accident
# severity using machine learning models and explainability tools.
#
# Research Questions:
#   RQ1: How do driver-related factors influence accident severity?
#         → Logistic Regression
#   RQ2: How do environmental factors affect accident severity?
#         → Random Forest
#   RQ3: What is the impact of temporal and weather conditions?
#         → Random Forest
#   RQ4: How do all factors interact to determine accident severity?
#         → Random Forest + SHAP + PDP + LIME
#
# Enhancements over base version:
#   - Data quality report with outlier detection
#   - EDA section with visualisations before modelling
#   - Label encoder mappings saved for interpretability
#   - Cross-validation added to all models
#   - Confusion matrix heatmaps for all models
#   - ROC-AUC score added to all models
#   - Feature importance plot saved to outputs/
#   - SHAP plots saved to outputs/
#   - LIME explanation added to RQ4
#   - Full summary report printed at end
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.inspection import PartialDependenceDisplay
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# Global plot style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 150, 'axes.titlesize': 13, 'axes.labelsize': 11})

# Store summary results across all RQs
summary_results = {}


# =============================================================================
# 1. LOAD & DATA QUALITY REPORT
# =============================================================================

def load_preprocess_data():
    """
    Load the road accident dataset, encode categorical columns, scale features,
    and apply SMOTE to handle class imbalance.

    Returns
    -------
    df            : DataFrame – Raw (encoded) dataset
    X_resampled   : np.ndarray – SMOTE-resampled feature matrix
    y_resampled   : np.ndarray – SMOTE-resampled target labels
    features      : list – Feature column names
    label_encoders: dict – Fitted LabelEncoder per categorical column
    scaler        : StandardScaler – Fitted scaler (for inverse transforms if needed)
    """
    df = pd.read_csv("Dataset/Road_Accident_Dataset.csv")

    print("=" * 60)
    print("  DATA QUALITY REPORT")
    print("=" * 60)
    print(f"  Shape           : {df.shape}")
    print(f"  Duplicate rows  : {df.duplicated().sum()}")
    print(f"  Missing values  : {df.isnull().sum().sum()}")

    missing = df.isnull().sum()
    if missing.any():
        print("\nMissing values per column:")
        print(missing[missing > 0].to_string())
    else:
        print("  No missing values found.")

    # --- Outlier detection (IQR) on numeric columns ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print("\nOutlier Detection (IQR method):")
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        n_out  = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        if n_out > 0:
            print(f"  {col:<35}: {n_out} outliers")

    print("\nClass Distribution (Accident Severity):")
    print(df['Accident Severity'].value_counts())

    # --- Encode categorical columns, save mappings ---
    cat_cols = [
        'Road Condition', 'Weather Conditions', 'Urban/Rural',
        'Driver Age Group', 'Driver Gender', 'Vehicle Condition', 'Accident Severity'
    ]
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"  Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Encode Time of Day if present
    if 'Time of Day' in df.columns and df['Time of Day'].dtype == object:
        le = LabelEncoder()
        df['Time of Day'] = le.fit_transform(df['Time of Day'])
        label_encoders['Time of Day'] = le

    features = [
        'Traffic Volume', 'Population Density', 'Speed Limit', 'Visibility Level',
        'Driver Alcohol Level', 'Driver Fatigue', 'Pedestrians Involved',
        'Number of Injuries', 'Number of Fatalities', 'Emergency Response Time',
        'Medical Cost', 'Economic Loss', 'Road Condition', 'Weather Conditions',
        'Urban/Rural', 'Driver Age Group', 'Driver Gender', 'Vehicle Condition'
    ]
    # Keep only features that exist in the dataframe
    features = [f for f in features if f in df.columns]

    target  = 'Accident Severity'
    X       = df[features]
    y       = df[target]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

    print(f"\nAfter SMOTE resampling: {X_resampled.shape[0]:,} samples")
    return df, X_resampled, y_resampled, features, label_encoders, scaler


df, X_resampled, y_resampled, all_features, label_encoders, scaler = load_preprocess_data()


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("\n" + "=" * 60)
print("  EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --- Class distribution ---
plt.figure(figsize=(7, 4))
df['Accident Severity'].value_counts().plot(kind='bar', color='steelblue', edgecolor='white')
plt.title('Accident Severity Class Distribution')
plt.xlabel('Severity Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/eda_severity_distribution.png")
plt.show()

# --- Numeric feature distributions ---
numeric_features = [
    'Traffic Volume', 'Speed Limit', 'Visibility Level',
    'Driver Alcohol Level', 'Emergency Response Time', 'Medical Cost'
]
numeric_features = [f for f in numeric_features if f in df.columns]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.flatten(), numeric_features):
    sns.histplot(df[col], kde=True, ax=ax, color='steelblue', edgecolor='white')
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
for ax in axes.flatten()[len(numeric_features):]:
    ax.set_visible(False)
plt.suptitle('Feature Distributions', fontsize=14)
plt.tight_layout()
plt.savefig("outputs/eda_feature_distributions.png")
plt.show()

# --- Correlation heatmap ---
corr_cols = [f for f in all_features if f in df.columns] + ['Accident Severity']
corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, linewidths=0.4, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig("outputs/eda_correlation_heatmap.png")
plt.show()

# --- Accident severity by driver alcohol level ---
if 'Driver Alcohol Level' in df.columns:
    plt.figure(figsize=(9, 5))
    sns.boxplot(x='Accident Severity', y='Driver Alcohol Level', data=df, palette='Set2')
    plt.title('Driver Alcohol Level by Accident Severity')
    plt.xlabel('Accident Severity')
    plt.ylabel('Driver Alcohol Level')
    plt.tight_layout()
    plt.savefig("outputs/eda_alcohol_by_severity.png")
    plt.show()


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name, feature_names=None, cv_scores=None):
    """
    Evaluate a trained classifier and print metrics.
    Generates and saves a confusion matrix heatmap.

    Parameters
    ----------
    model        : fitted sklearn model
    X_test       : np.ndarray – Test features
    y_test       : array-like – True labels
    model_name   : str – Label used in plot titles and filenames
    feature_names: list – Feature names (used for display only)
    cv_scores    : np.ndarray – Cross-validation R² scores (optional)

    Returns
    -------
    dict – Accuracy, F1, ROC-AUC (OVR), and CV mean
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro')
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except Exception:
        auc = None

    print(f"\n  Accuracy         : {acc:.4f}")
    print(f"  F1 Score (Macro) : {f1:.4f}")
    if auc:
        print(f"  ROC-AUC (OVR)    : {auc:.4f}")
    if cv_scores is not None:
        print(f"  CV Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
    plt.title(f'Confusion Matrix — {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_").replace("/", "_")
    plt.savefig(f"outputs/cm_{fname}.png")
    plt.show()

    return {'accuracy': acc, 'f1': f1, 'auc': auc,
            'cv_mean': cv_scores.mean() if cv_scores is not None else None}


# =============================================================================
# 4. RQ1 — DRIVER-RELATED FACTORS (LOGISTIC REGRESSION)
# =============================================================================

def run_rq1_logistic(df):
    """
    RQ1: How do driver-related factors influence accident severity?

    Uses Logistic Regression on: Driver Age Group, Driver Gender,
    Driver Alcohol Level, Driver Fatigue.

    Returns
    -------
    dict – Model evaluation metrics
    """
    print("\n" + "=" * 60)
    print("  RQ1 — Driver-Related Factors (Logistic Regression)")
    print("=" * 60)

    features = ['Driver Age Group', 'Driver Gender', 'Driver Alcohol Level', 'Driver Fatigue']
    features = [f for f in features if f in df.columns]

    X = StandardScaler().fit_transform(df[features])
    y = df['Accident Severity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    metrics = evaluate_model(model, X_test, y_test, "RQ1 Logistic Regression", features, cv_scores)

    # --- Coefficient importance plot ---
    coeff_df = pd.DataFrame({
        'Feature'    : features,
        'Coefficient': np.abs(model.coef_).mean(axis=0)
    }).sort_values('Coefficient')

    plt.figure(figsize=(8, 4))
    plt.barh(coeff_df['Feature'], coeff_df['Coefficient'], color='#3498db', edgecolor='white')
    plt.title('RQ1 — Driver Feature Importance (Logistic Regression Coefficients)')
    plt.xlabel('Mean |Coefficient|')
    plt.tight_layout()
    plt.savefig("outputs/rq1_feature_importance.png")
    plt.show()

    return metrics


summary_results['RQ1'] = run_rq1_logistic(df)


# =============================================================================
# 5. RQ2 — ENVIRONMENTAL FACTORS (RANDOM FOREST)
# =============================================================================

def run_rq2_random_forest(df):
    """
    RQ2: How do environmental factors affect accident severity?

    Uses Random Forest on: Traffic Volume, Population Density, Urban/Rural.

    Returns
    -------
    dict – Model evaluation metrics
    """
    print("\n" + "=" * 60)
    print("  RQ2 — Environmental Factors (Random Forest)")
    print("=" * 60)

    features = ['Traffic Volume', 'Population Density', 'Urban/Rural']
    features = [f for f in features if f in df.columns]

    X = StandardScaler().fit_transform(df[features])
    y = df['Accident Severity']

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, stratify=y_res, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_res, y_res, cv=cv, scoring='accuracy')

    metrics = evaluate_model(model, X_test, y_test, "RQ2 Random Forest", features, cv_scores)

    # --- Feature importance ---
    imp_df = pd.DataFrame({
        'Feature'   : features,
        'Importance': model.feature_importances_
    }).sort_values('Importance')

    plt.figure(figsize=(7, 4))
    plt.barh(imp_df['Feature'], imp_df['Importance'], color='#2ecc71', edgecolor='white')
    plt.title('RQ2 — Environmental Feature Importances (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig("outputs/rq2_feature_importance.png")
    plt.show()

    return metrics


summary_results['RQ2'] = run_rq2_random_forest(df)


# =============================================================================
# 6. RQ3 — TEMPORAL AND WEATHER IMPACT (RANDOM FOREST)
# =============================================================================

def run_rq3_random_forest(df):
    """
    RQ3: What is the impact of temporal and weather conditions on severity?

    Uses Random Forest on: Weather Conditions, Road Condition,
    Visibility Level, Time of Day.

    Returns
    -------
    dict – Model evaluation metrics
    """
    print("\n" + "=" * 60)
    print("  RQ3 — Temporal & Weather Factors (Random Forest)")
    print("=" * 60)

    features = ['Weather Conditions', 'Road Condition', 'Visibility Level', 'Time of Day']
    features = [f for f in features if f in df.columns]

    X = StandardScaler().fit_transform(df[features])
    y = df['Accident Severity']

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.3, stratify=y_res, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_res, y_res, cv=cv, scoring='accuracy')

    metrics = evaluate_model(model, X_test, y_test, "RQ3 Random Forest", features, cv_scores)

    # --- Feature importance ---
    imp_df = pd.DataFrame({
        'Feature'   : features,
        'Importance': model.feature_importances_
    }).sort_values('Importance')

    plt.figure(figsize=(7, 4))
    plt.barh(imp_df['Feature'], imp_df['Importance'], color='#e67e22', edgecolor='white')
    plt.title('RQ3 — Weather/Temporal Feature Importances (Random Forest)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig("outputs/rq3_feature_importance.png")
    plt.show()

    return metrics


summary_results['RQ3'] = run_rq3_random_forest(df)


# =============================================================================
# 7. RQ4 — OVERALL FACTOR INTERACTION (RANDOM FOREST + SHAP + PDP + LIME)
# =============================================================================

def run_rq4_random_forest(X_resampled, y_resampled, all_features):
    """
    RQ4: How do all factors interact to determine accident severity?

    Uses Random Forest on all features with SHAP, PDP, and LIME explanations.

    Parameters
    ----------
    X_resampled : np.ndarray – SMOTE-resampled feature matrix
    y_resampled : np.ndarray – SMOTE-resampled target labels
    all_features: list       – Feature names

    Returns
    -------
    dict – Model evaluation metrics
    """
    print("\n" + "=" * 60)
    print("  RQ4 — Overall Factor Interaction (Random Forest + SHAP + PDP + LIME)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3,
        stratify=y_resampled, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='accuracy')

    metrics = evaluate_model(model, X_test, y_test, "RQ4 Overall Random Forest",
                             all_features, cv_scores)

    # ── Feature Importance Plot ───────────────────────────────────────────────
    importances = model.feature_importances_
    indices     = np.argsort(importances)
    colors      = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(indices)))

    plt.figure(figsize=(10, 7))
    plt.barh(np.array(all_features)[indices], importances[indices],
             color=colors, edgecolor='white')
    plt.title('RQ4 — Overall Feature Importances (Random Forest)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig("outputs/rq4_feature_importance.png")
    plt.show()

    # ── SHAP Summary Plot ─────────────────────────────────────────────────────
    print("\nGenerating SHAP explanations (this may take a moment)...")
    explainer   = shap.TreeExplainer(model)
    shap_sample = X_test[:500]
    shap_values = explainer.shap_values(shap_sample)

    # SHAP bar summary (mean |SHAP| per feature)
    shap.summary_plot(shap_values, shap_sample, feature_names=all_features,
                      plot_type='bar', show=False)
    plt.title('SHAP Mean Feature Importance (RQ4)')
    plt.tight_layout()
    plt.savefig("outputs/rq4_shap_bar.png")
    plt.show()

    # SHAP beeswarm plot
    shap.summary_plot(shap_values, shap_sample, feature_names=all_features, show=False)
    plt.title('SHAP Beeswarm — Feature Impact on Severity (RQ4)')
    plt.tight_layout()
    plt.savefig("outputs/rq4_shap_beeswarm.png")
    plt.show()

    # ── Partial Dependence Plots ──────────────────────────────────────────────
    pdp_features = [f for f in ['Driver Alcohol Level', 'Visibility Level', 'Speed Limit']
                    if f in all_features]
    pdp_indices  = [all_features.index(f) for f in pdp_features]

    if pdp_indices:
        PartialDependenceDisplay.from_estimator(
            model, X_test[:500], pdp_indices,
            feature_names=all_features, grid_resolution=20
        )
        plt.suptitle('Partial Dependence Plots — Key Features (RQ4)', fontsize=13)
        plt.tight_layout()
        plt.savefig("outputs/rq4_pdp.png")
        plt.show()

    # ── LIME Explanation ──────────────────────────────────────────────────────
    print("\nGenerating LIME explanation for one sample...")
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names  = all_features,
        class_names    = [str(c) for c in sorted(np.unique(y_resampled))],
        discretize_continuous = True,
        random_state   = 42
    )

    # Explain the first test sample
    lime_exp = lime_explainer.explain_instance(
        X_test[0], model.predict_proba, num_features=10
    )
    lime_exp.as_pyplot_figure()
    plt.title('LIME Explanation — Sample 0 (RQ4)')
    plt.tight_layout()
    plt.savefig("outputs/rq4_lime_explanation.png")
    plt.show()

    print("\nTop LIME features for sample 0:")
    for feat, weight in lime_exp.as_list():
        print(f"  {feat:<45} weight={weight:.4f}")

    return metrics


summary_results['RQ4'] = run_rq4_random_forest(X_resampled, y_resampled, all_features)


# =============================================================================
# 8. FINAL SUMMARY REPORT
# =============================================================================

print("\n" + "=" * 60)
print("  FINAL MODEL SUMMARY REPORT")
print("=" * 60)
print(f"  {'RQ':<6} {'Model':<30} {'Accuracy':>9} {'F1 (Macro)':>11} {'ROC-AUC':>9} {'CV Acc':>8}")
print("  " + "─" * 75)

model_names = {
    'RQ1': 'Logistic Regression (Driver)',
    'RQ2': 'Random Forest (Environment)',
    'RQ3': 'Random Forest (Weather/Time)',
    'RQ4': 'Random Forest (All Features)',
}

for rq, metrics in summary_results.items():
    acc  = f"{metrics['accuracy']:.4f}"
    f1   = f"{metrics['f1']:.4f}"
    auc  = f"{metrics['auc']:.4f}" if metrics['auc'] else "N/A"
    cv   = f"{metrics['cv_mean']:.4f}" if metrics['cv_mean'] else "N/A"
    print(f"  {rq:<6} {model_names[rq]:<30} {acc:>9} {f1:>11} {auc:>9} {cv:>8}")

print("\n  Output plots saved to: outputs/")
print("=" * 60)
