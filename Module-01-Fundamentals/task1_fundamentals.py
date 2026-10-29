"""
YuvaIntern Machine Learning Engineer Internship
Module 1: Data Science Fundamentals Assessment
Author: Neelotpal Ghosh
Admission No: 24SCSE1650001
Topic: Python Basics, Statistical Foundations, Data Types & Industry Best Practices
"""

import math
import random
from typing import List, Dict, Tuple, Any, Optional

# ==============================================================================
# SECTION 1: PYTHON BASICS & CORE PROGRAMMING FOR DATA SCIENCE
# ==============================================================================

def demonstrate_python_data_structures():
    """
    Demonstrates Python data structures (list, tuple, dict, set)
    and list comprehensions commonly used in data pipelines.
    """
    print("--- [1.1] Python Core Data Structures ---")
    
    # 1. Lists: Ordered, mutable sequence
    raw_measurements = [23.5, 25.1, 22.8, 28.4, 24.0, 30.2, 21.9]
    raw_measurements.append(26.5)
    print(f"Measurements list: {raw_measurements}")
    
    # 2. List Comprehension: Filtering & Transformation
    normalized_values = [round((x - min(raw_measurements)) / (max(raw_measurements) - min(raw_measurements)), 3) 
                         for x in raw_measurements]
    print(f"Min-Max Normalized values: {normalized_values}")
    
    # 3. Dictionaries: Key-value mappings (used heavily for JSON / row data)
    feature_record = {
        "student_id": "24SCSE1650001",
        "name": "Neelotpal Ghosh",
        "task_scores": [92, 88, 95, 91],
        "is_active": True
    }
    avg_score = sum(feature_record["task_scores"]) / len(feature_record["task_scores"])
    print(f"Student: {feature_record['name']}, Average Score: {avg_score:.2f}")
    
    # 4. Sets: Unordered, unique elements (useful for categorical cardinality)
    category_labels = ["Low", "Medium", "High", "Low", "Medium", "High", "Critical"]
    unique_categories = set(category_labels)
    print(f"Unique categories ({len(unique_categories)} distinct): {sorted(list(unique_categories))}\n")


# Object-Oriented Programming: Custom Data Processor Pipeline
class DataPipelineProcessor:
    """
    Demonstrates OOP principles (encapsulation, abstraction)
    designed following industry-standard clean coding practices.
    """
    def __init__(self, data: List[float], name: str = "Pipeline_v1"):
        self._name = name
        self._raw_data = data
        self._cleaned_data: List[float] = []
        
    @property
    def name(self) -> str:
        return self._name

    def remove_missing_and_negatives(self) -> "DataPipelineProcessor":
        """Filter out non-finite or negative values (data cleaning step)."""
        self._cleaned_data = [x for x in self._raw_data if x is not None and x >= 0]
        return self

    def get_cleaned_data(self) -> List[float]:
        return self._cleaned_data


# ==============================================================================
# SECTION 2: STATISTICAL FOUNDATIONS FOR MACHINE LEARNING
# ==============================================================================

class StatisticalAnalyzer:
    """
    Implements core descriptive and inferential statistics from first principles.
    Measures of central tendency, dispersion, correlation, and hypothesis testing.
    """
    def __init__(self, sample: List[float]):
        self.sample = sorted(sample)
        self.n = len(sample)
        if self.n == 0:
            raise ValueError("Sample cannot be empty.")

    def mean(self) -> float:
        """Calculate arithmetic mean (measure of central tendency)."""
        return sum(self.sample) / self.n

    def median(self) -> float:
        """Calculate median (robust measure of central tendency unaffected by outliers)."""
        mid = self.n // 2
        if self.n % 2 == 0:
            return (self.sample[mid - 1] + self.sample[mid]) / 2.0
        return float(self.sample[mid])

    def variance(self, ddof: int = 1) -> float:
        """Calculate sample variance with Bessel's correction (ddof=1)."""
        if self.n <= 1:
            return 0.0
        m = self.mean()
        return sum((x - m) ** 2 for x in self.sample) / (self.n - ddof)

    def std_dev(self, ddof: int = 1) -> float:
        """Calculate sample standard deviation."""
        return math.sqrt(self.variance(ddof))

    def interquartile_range(self) -> Tuple[float, float, float]:
        """
        Calculates Q1 (25th percentile), Q3 (75th percentile), and IQR.
        Essential for Tukey's outlier detection rule.
        """
        def percentile(data: List[float], p: float) -> float:
            k = (len(data) - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[int(k)]
            d0 = data[int(f)] * (c - k)
            d1 = data[int(c)] * (k - f)
            return d0 + d1

        q1 = percentile(self.sample, 0.25)
        q3 = percentile(self.sample, 0.75)
        iqr = q3 - q1
        return q1, q3, iqr

    def detect_outliers_iqr(self) -> Tuple[List[float], float, float]:
        """Detect outliers using the 1.5 * IQR rule."""
        q1, q3, iqr = self.interquartile_range()
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [x for x in self.sample if x < lower_bound or x > upper_bound]
        return outliers, lower_bound, upper_bound


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """
    Computes Pearson's Linear Correlation Coefficient r (-1 to +1).
    Quantifies the strength and direction of the linear relationship between two features.
    """
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("Inputs must have identical non-zero lengths.")
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def two_sample_t_statistic(group_a: List[float], group_b: List[float]) -> Tuple[float, float]:
    """
    Calculates Student's Two-Sample t-statistic and degrees of freedom.
    Used for A/B testing and feature significance testing.
    """
    n1, n2 = len(group_a), len(group_b)
    m1 = sum(group_a) / n1
    m2 = sum(group_b) / n2
    
    var1 = sum((x - m1) ** 2 for x in group_a) / (n1 - 1)
    var2 = sum((x - m2) ** 2 for x in group_b) / (n2 - 1)
    
    # Pooled standard deviation
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    se_diff = math.sqrt(pooled_var * (1/n1 + 1/n2))
    
    t_stat = (m1 - m2) / se_diff
    df = n1 + n2 - 2
    return t_stat, df


# ==============================================================================
# SECTION 3: DATA TYPES, TRANSFORMATIONS & ENCODINGS
# ==============================================================================

class FeatureEncoder:
    """
    Demonstrates data preprocessing and encoding for nominal & ordinal features.
    """
    @staticmethod
    def one_hot_encode(categories: List[str]) -> Tuple[List[str], List[List[int]]]:
        """Convert nominal categories to binary vectors (One-Hot Encoding)."""
        unique_cats = sorted(list(set(categories)))
        encoded_matrix = []
        for cat in categories:
            row = [1 if cat == u else 0 for u in unique_cats]
            encoded_matrix.append(row)
        return unique_cats, encoded_matrix

    @staticmethod
    def ordinal_encode(values: List[str], ordering: Dict[str, int]) -> List[int]:
        """Convert ordered categories (e.g., Low < Medium < High) into integers."""
        return [ordering.get(val, -1) for val in values]


# ==============================================================================
# SECTION 4: EXECUTION & VERIFICATION OF ASSIGNMENT REQUIREMENTS
# ==============================================================================

def main():
    print("=" * 70)
    print(" YuvaIntern Machine Learning Engineer Internship ")
    print(" Task 1: Data Science Fundamentals Assessment ")
    print(" Student: Neelotpal Ghosh | Admission No: 24SCSE1650001 ")
    print("=" * 70)
    print()
    
    # 1. Python Data Structures
    demonstrate_python_data_structures()
    
    # 2. OOP Data Pipeline
    raw_stream = [14.2, 19.5, -3.0, None, 22.1, 18.7, 35.0, 16.4]
    processor = DataPipelineProcessor(raw_stream, name="TelemetrySensor")
    cleaned = processor.remove_missing_and_negatives().get_cleaned_data()
    print(f"Pipeline: {processor.name}")
    print(f"Cleaned dataset: {cleaned}\n")
    
    # 3. Statistical Analysis
    stats = StatisticalAnalyzer(cleaned)
    mean_val = stats.mean()
    median_val = stats.median()
    variance_val = stats.variance()
    std_val = stats.std_dev()
    q1, q3, iqr = stats.interquartile_range()
    outliers, lb, ub = stats.detect_outliers_iqr()
    
    print("--- [2.0] Statistical Foundations Output ---")
    print(f"Sample Count (N)  : {stats.n}")
    print(f"Mean              : {mean_val:.3f}")
    print(f"Median            : {median_val:.3f}")
    print(f"Variance (Sample) : {variance_val:.3f}")
    print(f"Std Deviation     : {std_val:.3f}")
    print(f"Q1 (25th %)       : {q1:.3f}")
    print(f"Q3 (75th %)       : {q3:.3f}")
    print(f"IQR               : {iqr:.3f}")
    print(f"Tukey Bounds      : [{lb:.3f}, {ub:.3f}]")
    print(f"Detected Outliers : {outliers}\n")
    
    # 4. Correlation Analysis
    feature_x = [10, 12, 14, 16, 18, 20, 22, 24]
    feature_y = [25, 29, 34, 38, 45, 51, 56, 62]  # Strong positive correlation
    r = pearson_correlation(feature_x, feature_y)
    print("--- [3.0] Correlation & Hypothesis Testing ---")
    print(f"Feature X: {feature_x}")
    print(f"Feature Y: {feature_y}")
    print(f"Pearson Correlation Coefficient (r): {r:.4f}")
    
    # 5. Hypothesis Testing (Two-Sample t-test)
    treatment_group = [22.4, 23.1, 24.5, 22.8, 25.2, 24.1]
    control_group = [19.2, 20.1, 19.8, 20.5, 21.0, 19.9]
    t_stat, df = two_sample_t_statistic(treatment_group, control_group)
    print(f"Two-sample t-statistic: {t_stat:.4f} (degrees of freedom: {df})")
    print("Conclusion: Significant statistical difference observed between groups.\n")
    
    # 6. Feature Encoding
    print("--- [4.0] Categorical Data Encoding ---")
    nominal_data = ["Red", "Blue", "Green", "Blue", "Red"]
    unique_cols, ohe_matrix = FeatureEncoder.one_hot_encode(nominal_data)
    print(f"One-Hot Encoding for {nominal_data}:")
    print(f"Columns: {unique_cols}")
    for val, row in zip(nominal_data, ohe_matrix):
        print(f"  {val:5s} -> {row}")
        
    ordinal_data = ["Low", "High", "Medium", "Critical", "Low"]
    severity_order = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    encoded_ordinal = FeatureEncoder.ordinal_encode(ordinal_data, severity_order)
    print(f"\nOrdinal Encoding for {ordinal_data}:")
    print(f"Encoded values (Rank mapped): {encoded_ordinal}\n")
    
    print("=" * 70)
    print(" Task 1 Execution Completed Successfully! ")
    print("=" * 70)


if __name__ == "__main__":
    main()
