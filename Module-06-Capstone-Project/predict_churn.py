"""
ChurnGuard Real-Time Inference Script
Author: Neelotpal Ghosh
Admission No: 24SCSE1650001
YuvaIntern Machine Learning Engineer Capstone Project
"""

import os
import joblib
import pandas as pd

def score_incoming_customer(payload: dict, model_path: str = "churnguard_model.joblib"):
    """
    Loads serialized ChurnGuard pipeline artifact and produces
    real-time churn probability, risk classification, and retention recommendation.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")
        
    pipeline = joblib.load(model_path)
    input_df = pd.DataFrame([payload])
    
    churn_prob = pipeline.predict_proba(input_df)[0, 1]
    churn_pred = int(churn_prob >= 0.40) # Business optimal threshold
    
    risk_level = "HIGH RISK" if churn_prob >= 0.50 else ("MEDIUM RISK" if churn_prob >= 0.25 else "LOW RISK")
    
    recommendation = (
        "Immediate VIP Customer Success Intervention & Annual Contract Migration Discount (15%)"
        if risk_level == "HIGH RISK" else
        ("Automated Onboarding Engagement Check & Feature Usage Walkthrough" if risk_level == "MEDIUM RISK"
         else "Standard Monthly Communication; Customer Healthy")
    )
    
    return {
        "CustomerID": payload.get("CustomerID", "N/A"),
        "ChurnProbability": round(float(churn_prob), 4),
        "PredictedChurn": churn_pred,
        "RiskTier": risk_level,
        "ActionableRecommendation": recommendation
    }

if __name__ == "__main__":
    sample_customer = {
        "CustomerID": "CG-9999",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "Tenure": 3,
        "MonthlyCharges": 98.50,
        "TotalCharges": 295.50,
        "SupportCalls": 4,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "TechSupport": "No",
        "OnlineSecurity": "No",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(base_dir, "churnguard_model.joblib")
    
    if os.path.exists(model_file):
        result = score_incoming_customer(sample_customer, model_file)
        print("=" * 60)
        print(" CHURNGUARD REAL-TIME INFERENCE RESULT ")
        print("=" * 60)
        for k, v in result.items():
            print(f"  {k:26s}: {v}")
        print("=" * 60)
    else:
        print("Model file not found. Train the model first via task6_capstone_pipeline.py.")
