"""
YuvaIntern Machine Learning Engineer Internship
Module 2: Hands-On Data Lab Implementation
Author: Neelotpal Ghosh
Admission No: 24SCSE1650001
Topic: Environment Setup, Data Manipulation, Statistical Visualization & Basic Analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for automated generation
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

def generate_sample_dataset(n_samples: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic enterprise workforce analytics dataset with numerical,
    categorical, and intentionally introduced missing values for data lab exercises.
    """
    np.random.seed(random_seed)
    
    departments = ["Engineering", "Data Science", "Product", "Marketing", "Sales"]
    education_levels = ["Bachelor's", "Master's", "PhD"]
    
    dept_choices = np.random.choice(departments, size=n_samples, p=[0.35, 0.20, 0.15, 0.15, 0.15])
    edu_choices = np.random.choice(education_levels, size=n_samples, p=[0.55, 0.35, 0.10])
    
    # Generate realistic correlated features
    years_exp = np.clip(np.random.gamma(shape=3.0, scale=2.0, size=n_samples), 0.5, 25.0)
    age = np.clip(22 + years_exp + np.random.normal(loc=0, scale=2.5, size=n_samples), 21, 65)
    
    # Department base salary multipliers
    base_salaries = {
        "Engineering": 65000,
        "Data Science": 72000,
        "Product": 68000,
        "Marketing": 52000,
        "Sales": 50000
    }
    
    salaries = []
    for dept, exp, edu in zip(dept_choices, years_exp, edu_choices):
        base = base_salaries[dept]
        edu_boost = 1.0 if edu == "Bachelor's" else (1.18 if edu == "Master's" else 1.35)
        exp_boost = exp * 4200
        noise = np.random.normal(0, 5000)
        salaries.append(max(35000, round((base + exp_boost + noise) * edu_boost, 2)))
        
    performance_scores = np.clip(np.random.normal(loc=78, scale=11, size=n_samples), 45, 100)
    hours_weekly = np.clip(np.random.normal(loc=41.5, scale=4.2, size=n_samples), 30, 60)
    
    df = pd.DataFrame({
        "EmployeeID": [f"EMP-{1000 + i}" for i in range(n_samples)],
        "Department": dept_choices,
        "EducationLevel": edu_choices,
        "Age": np.round(age, 1),
        "YearsExperience": np.round(years_exp, 1),
        "AnnualSalary": salaries,
        "PerformanceScore": np.round(performance_scores, 1),
        "WeeklyHours": np.round(hours_weekly, 1),
        "RemoteStatus": np.random.choice(["Remote", "Hybrid", "On-Site"], size=n_samples, p=[0.4, 0.4, 0.2])
    })
    
    # Introduce controlled missing values to demonstrate data cleaning
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices[:len(missing_indices)//2], "YearsExperience"] = np.nan
    df.loc[missing_indices[len(missing_indices)//2:], "PerformanceScore"] = np.nan
    
    return df


def execute_data_manipulation_exercises(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carries out structured data manipulation exercises using Pandas:
    cleaning, imputation, multi-level aggregation, reshaping, and feature engineering.
    """
    print("=" * 60)
    print(" EXERCISE 1: DATA MANIPULATION & WRANGLING ")
    print("=" * 60)
    
    # 1. Inspect initial shape and missingness
    print(f"Original shape: {df.shape}")
    print("Missing values per column:\n", df.isnull().sum()[df.isnull().sum() > 0])
    
    # 2. Imputation: Group-based median imputation
    df_clean = df.copy()
    df_clean["YearsExperience"] = df_clean.groupby("Department")["YearsExperience"].transform(
        lambda x: x.fillna(x.median())
    )
    df_clean["PerformanceScore"] = df_clean.groupby("Department")["PerformanceScore"].transform(
        lambda x: x.fillna(x.median())
    )
    print("\nMissing values after imputation:\n", df_clean.isnull().sum().to_dict())
    
    # 3. Feature Engineering
    df_clean["SalaryPerYearExp"] = np.round(df_clean["AnnualSalary"] / np.maximum(df_clean["YearsExperience"], 1.0), 2)
    df_clean["PerformanceEfficiency"] = np.round(df_clean["PerformanceScore"] / df_clean["WeeklyHours"], 3)
    
    # 4. Multi-level Aggregation by Department
    dept_summary = df_clean.groupby("Department").agg(
        TotalEmployees=("EmployeeID", "count"),
        MeanSalary=("AnnualSalary", "mean"),
        MedianSalary=("AnnualSalary", "median"),
        MeanExperience=("YearsExperience", "mean"),
        MeanPerformance=("PerformanceScore", "mean"),
        StdSalary=("AnnualSalary", "std")
    ).round(2).sort_values("MeanSalary", ascending=False)
    
    print("\n--- Departmental Aggregate Summary ---")
    print(dept_summary)
    
    # 5. Reshaping / Pivot Table
    salary_pivot = df_clean.pivot_table(
        values="AnnualSalary",
        index="Department",
        columns="EducationLevel",
        aggfunc="mean"
    ).round(2)
    
    print("\n--- Salary Pivot Table (Department x Education) ---")
    print(salary_pivot)
    
    return df_clean, dept_summary, salary_pivot


def generate_visualizations(df: pd.DataFrame, output_dir: str):
    """
    Generates 4 publication-quality statistical charts using Matplotlib & Seaborn.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print(" EXERCISE 2: STATISTICAL VISUALIZATIONS ")
    print("=" * 60)
    
    # 1. Distribution Plot: Annual Salary with KDE
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    sns.histplot(df["AnnualSalary"], kde=True, color="#102C57", bins=30, ax=ax, edgecolor="white", alpha=0.85)
    mean_sal = df["AnnualSalary"].mean()
    median_sal = df["AnnualSalary"].median()
    ax.axvline(mean_sal, color="#E63946", linestyle="--", linewidth=1.8, label=f"Mean: ${mean_sal:,.0f}")
    ax.axvline(median_sal, color="#35A29F", linestyle="-", linewidth=1.8, label=f"Median: ${median_sal:,.0f}")
    ax.set_title("Figure 1: Salary Distribution with Kernel Density Estimation (KDE)", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Annual Salary (USD)", fontsize=10, weight="bold")
    ax.set_ylabel("Frequency Count", fontsize=10, weight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
    plt.tight_layout()
    p1 = os.path.join(output_dir, "fig1_salary_distribution.png")
    fig.savefig(p1)
    plt.close()
    print(f"Saved: {p1}")
    
    # 2. Categorical Comparison: Boxplot by Department and Education
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.boxplot(
        data=df,
        x="Department",
        y="AnnualSalary",
        hue="EducationLevel",
        palette=["#35A29F", "#102C57", "#F4A261"],
        ax=ax,
        width=0.6,
        linewidth=1.2,
        fliersize=3
    )
    ax.set_title("Figure 2: Salary Distribution Across Departments & Education Levels", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Department", fontsize=10, weight="bold")
    ax.set_ylabel("Annual Salary (USD)", fontsize=10, weight="bold")
    ax.legend(title="Education Level", frameon=True, facecolor="white")
    plt.tight_layout()
    p2 = os.path.join(output_dir, "fig2_salary_by_department.png")
    fig.savefig(p2)
    plt.close()
    print(f"Saved: {p2}")
    
    # 3. Scatter Plot: Experience vs Salary colored by Performance
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    scatter = ax.scatter(
        df["YearsExperience"],
        df["AnnualSalary"],
        c=df["PerformanceScore"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
        s=45
    )
    # Regression trend line
    m, b = np.polyfit(df["YearsExperience"], df["AnnualSalary"], 1)
    x_vals = np.linspace(df["YearsExperience"].min(), df["YearsExperience"].max(), 100)
    ax.plot(x_vals, m * x_vals + b, color="#E63946", linewidth=2.0, label=f"Trend: Slope = ${m:,.0f}/yr")
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Performance Score", fontsize=10, weight="bold")
    ax.set_title("Figure 3: Experience vs Annual Salary (Color Coded by Performance)", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Years of Experience", fontsize=10, weight="bold")
    ax.set_ylabel("Annual Salary (USD)", fontsize=10, weight="bold")
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    p3 = os.path.join(output_dir, "fig3_experience_vs_salary.png")
    fig.savefig(p3)
    plt.close()
    print(f"Saved: {p3}")
    
    # 4. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
    num_cols = ["Age", "YearsExperience", "AnnualSalary", "PerformanceScore", "WeeklyHours", "SalaryPerYearExp"]
    corr_matrix = df[num_cols].corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Pearson Correlation'}
    )
    ax.set_title("Figure 4: Pearson Correlation Matrix of Numeric Metrics", fontsize=13, weight="bold", pad=12)
    plt.tight_layout()
    p4 = os.path.join(output_dir, "fig4_correlation_heatmap.png")
    fig.savefig(p4)
    plt.close()
    print(f"Saved: {p4}")


def main():
    print("=" * 70)
    print(" YuvaIntern Machine Learning Engineer Internship ")
    print(" Task 2: Hands-On Data Lab Implementation ")
    print(" Student: Neelotpal Ghosh | Admission No: 24SCSE1650001 ")
    print("=" * 70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    
    # 1. Environment and version audit
    print("\n[Environment Audit]")
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"NumPy Version     : {np.__version__}")
    print(f"Pandas Version    : {pd.__version__}")
    print(f"Matplotlib Version: {matplotlib.__version__}")
    print(f"Seaborn Version   : {sns.__version__}")
    
    # 2. Dataset Synthesis & Loading
    raw_df = generate_sample_dataset(n_samples=500, random_seed=42)
    csv_path = os.path.join(base_dir, "workforce_lab_data.csv")
    raw_df.to_csv(csv_path, index=False)
    print(f"\nGenerated lab dataset saved to: {csv_path}")
    
    # 3. Data manipulation exercises
    clean_df, dept_summary, pivot = execute_data_manipulation_exercises(raw_df)
    
    # 4. Visualizations
    generate_visualizations(clean_df, figures_dir)
    
    print("\n" + "=" * 70)
    print(" Task 2 Data Lab Execution Completed Successfully! ")
    print("=" * 70)

if __name__ == "__main__":
    main()
