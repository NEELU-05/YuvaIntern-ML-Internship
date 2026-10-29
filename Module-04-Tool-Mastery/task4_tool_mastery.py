"""
YuvaIntern Machine Learning Engineer Internship
Module 4: Data Science Tool Mastery Project
Author: Neelotpal Ghosh
Admission No: 24SCSE1650001
Topic: Industry-Standard Toolkit Mastery (NumPy, Pandas, Scikit-Learn Pipelines, Matplotlib, Seaborn)
Portfolio Showcase: End-to-End Supervised Machine Learning Benchmark & Feature Importance Pipeline
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Arial'

def generate_telecom_data(n_samples: int = 1200, random_seed: int = 42) -> pd.DataFrame:
    """Generates benchmark dataset for modeling."""
    np.random.seed(random_seed)
    
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    internets = np.random.choice(["Fiber optic", "DSL", "No"], size=n_samples, p=[0.45, 0.35, 0.20])
    tech_supports = np.random.choice(["Yes", "No", "No internet service"], size=n_samples, p=[0.30, 0.50, 0.20])
    payments = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_samples)
    seniors = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    
    tenures = []
    charges = []
    churn_labels = []
    
    for c, i, s in zip(contracts, internets, seniors):
        t = int(np.clip(np.random.exponential(14.0) if c == "Month-to-month" else np.random.normal(45, 12), 1, 72))
        tenures.append(t)
        
        ch = 92.0 + np.random.normal(0, 10) if i == "Fiber optic" else (58.0 + np.random.normal(0, 8) if i == "DSL" else 21.0 + np.random.normal(0, 3))
        charges.append(round(float(np.clip(ch, 18, 118)), 2))
        
        logit = -1.5 + (1.3 if c == "Month-to-month" else -1.6) + (0.8 if i == "Fiber optic" else -0.9) - (0.04 * t) + (0.015 * ch)
        prob = 1.0 / (1.0 + np.exp(-logit))
        churn_labels.append(1 if np.random.rand() < prob else 0)
        
    total_charges = [round(t * m + np.random.normal(0, 5), 2) for t, m in zip(tenures, charges)]
    
    df = pd.DataFrame({
        "SeniorCitizen": seniors,
        "Tenure": tenures,
        "MonthlyCharges": charges,
        "TotalCharges": total_charges,
        "Contract": contracts,
        "InternetService": internets,
        "TechSupport": tech_supports,
        "PaymentMethod": payments,
        "Churn": churn_labels
    })
    return df


def build_and_evaluate_models(df: pd.DataFrame, output_dir: str):
    """
    Builds Scikit-learn Pipelines with ColumnTransformer and benchmarks:
    1. Logistic Regression (Linear Baseline)
    2. Decision Tree Classifier (Non-linear Baseline)
    3. Random Forest Classifier (Ensemble Bagging)
    4. Gradient Boosting Classifier (Ensemble Boosting)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    numeric_features = ["Tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = ["SeniorCitizen", "Contract", "InternetService", "TechSupport", "PaymentMethod"]
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }
    
    results = []
    trained_pipelines = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("=" * 65)
    print(" MODEL BENCHMARKING WITH SCIKIT-LEARN PIPELINES ")
    print("=" * 65)
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        
        # 5-fold cross-validation on ROC-AUC
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "CV_AUC_Mean": cv_scores.mean(),
            "CV_AUC_Std": cv_scores.std()
        })
        
        print(f"[{name}]")
        print(f"  Accuracy : {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
        print(f"  5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})\n")
        
    results_df = pd.DataFrame(results).round(4)
    
    # ---------------------------------------------------------
    # GENERATE PUBLICATION-GRADE VISUALIZATIONS
    # ---------------------------------------------------------
    print("=" * 65)
    print(" GENERATING PORTFOLIO VISUALIZATIONS ")
    print("=" * 65)
    
    # 1. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    metrics_melted = results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                                     var_name="Metric", value_name="Score")
    sns.barplot(data=metrics_melted, x="Model", y="Score", hue="Metric", palette="Blues_r", ax=ax)
    ax.set_title("Figure 1: Comprehensive Model Performance Comparison Across Evaluation Metrics", fontsize=12, weight="bold", pad=12)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Score", fontsize=10, weight="bold")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    p1 = os.path.join(output_dir, "fig1_model_comparison_metrics.png")
    fig.savefig(p1)
    plt.close()
    print(f"Saved: {p1}")
    
    # 2. Side-by-Side Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    for idx, (m_name, ax) in enumerate(zip(["Logistic Regression", "Gradient Boosting"], axes)):
        pipe = trained_pipelines[m_name]
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Retained (0)", "Churned (1)"], yticklabels=["Retained (0)", "Churned (1)"])
        ax.set_title(f"Figure 2{chr(97+idx)}: Confusion Matrix — {m_name}", fontsize=11, weight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10, weight="bold")
        ax.set_ylabel("True Label", fontsize=10, weight="bold")
    plt.tight_layout()
    p2 = os.path.join(output_dir, "fig2_confusion_matrices.png")
    fig.savefig(p2)
    plt.close()
    print(f"Saved: {p2}")
    
    # 3. Multi-Model ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for name, pipe in trained_pipelines.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2.0)
        
    ax.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.500)", linewidth=1.5)
    ax.set_title("Figure 3: Receiver Operating Characteristic (ROC) Curves", fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, weight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10, weight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white")
    plt.tight_layout()
    p3 = os.path.join(output_dir, "fig3_roc_auc_curves.png")
    fig.savefig(p3)
    plt.close()
    print(f"Saved: {p3}")
    
    # 4. Feature Importance (Random Forest Gini Importance)
    rf_pipe = trained_pipelines["Random Forest"]
    rf_model = rf_pipe.named_steps["classifier"]
    
    # Extract feature names from ColumnTransformer
    cat_encoder = rf_pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_cols = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_cols
    
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Figure 4: Top 10 Features Driving Predictions (Random Forest Gini Importance)", fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("Mean Impurity Decrease (Feature Importance)", fontsize=10, weight="bold")
    ax.set_ylabel("Engineered Feature", fontsize=10, weight="bold")
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.3f}", (p.get_width() + 0.005, p.get_y() + p.get_height() / 2.),
                    va='center', fontsize=9.5, color='#102C57', weight='bold')
    ax.set_xlim(0, max(feat_df["Importance"]) * 1.18)
    plt.tight_layout()
    p4 = os.path.join(output_dir, "fig4_feature_importance.png")
    fig.savefig(p4)
    plt.close()
    print(f"Saved: {p4}")
    
    return results_df, feat_df


def main():
    print("=" * 70)
    print(" YuvaIntern Machine Learning Engineer Internship ")
    print(" Task 4: Data Science Tool Mastery Project ")
    print(" Student: Neelotpal Ghosh | Admission No: 24SCSE1650001 ")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    
    # Generate & persist dataset
    raw_df = generate_telecom_data(n_samples=1200, random_seed=42)
    csv_path = os.path.join(base_dir, "benchmark_modeling_data.csv")
    raw_df.to_csv(csv_path, index=False)
    print(f"\nPersisted benchmark dataset to: {csv_path}")
    
    # Train, benchmark & visualize
    results_df, feat_df = build_and_evaluate_models(raw_df, figures_dir)
    
    print("\n--- Final Model Benchmark Leaderboard ---")
    print(results_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print(" Task 4 Tool Mastery Execution Completed Successfully! ")
    print("=" * 70)

if __name__ == "__main__":
    main()
