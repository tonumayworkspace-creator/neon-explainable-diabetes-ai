# 🧾 Model Card — Neon Explainable Diabetes Risk Model

## 1. Model Details

**Model Name:** Neon Explainable Diabetes Risk Model  
**Version:** 1.0  
**Algorithm:** Scikit-Learn Classifier (trained using pipeline)  
**Explainability:** SHAP value explainability  
**Intended Platform:** Streamlit Web Application  

### Stakeholders
- Data Scientists
- Healthcare Researchers
- ML Engineers
- Compliance / Model Risk Governance Teams

---

## 2. Intended Use

This model estimates the **probability of diabetes risk** based on structured patient health attributes and provides **explainable reasoning** behind predictions.

### Primary intended uses
- educational demonstration of Explainable AI
- healthcare AI portfolio project
- research prototyping
- Responsible AI showcase

### Not intended for
- clinical decision-making
- medical diagnosis
- emergency decision systems

---

## 3. Training Data

- Tabular patient-level dataset
- Contains health attributes such as BMI, glucose, age, etc.
- Labels: diabetes diagnosis status (binary)

### Sensitive attributes
The system may contain variables correlated with:
- age
- gender
- lifestyle indicators

The dashboard explicitly features:
- fairness analysis
- transparency scoring

---

## 4. Model Performance

Evaluation metrics considered:

- ROC-AUC  
- Precision / Recall  
- F1-Score  
- Calibration behaviour  

Performance can vary by subpopulation — fairness view provides visualization.

---

## 5. Explainability

The model includes:

- SHAP value explanations
- Waterfall plots
- Top-feature contribution lists
- Local instance explanations
- What-If counterfactual simulation

Example interpretation:

> “Elevated glucose and BMI are key contributors to high estimated diabetes risk.”

The goal is **human-interpretable output** for non-technical users.

---

## 6. Fairness & Bias

### Supported functionality
- group-wise risk comparison
- transparency score
- consent acknowledgement
- bias signal visualization

### Identified risks
- training data may be imbalanced
- under-representation possible
- spurious correlations may exist

This model **does not guarantee removal of bias** but provides **visibility into it.**

---

## 7. Ethical Considerations

- Not for medical use
- Never replace clinician judgement
- Predictions may be wrong
- Sensitive attribute bias possible
- Explainability may be imperfect

The UI includes:
- disclaimer
- consent acknowledgement
- governance language

---

## 8. Caveats and Recommendations

Users should:

- avoid deploying this system in clinical settings
- avoid using outputs for diagnosis
- review subgroup fairness before reuse
- supplement with clinical knowledge

Future improvements:

- larger clinical datasets
- stronger fairness constraints
- calibrated medical thresholds
- clinician-validated interpretations

---

## 9. Contact

**Author:** Tonumay Bhattacharya  
**Focus Areas:** Data Science | ML | Explainable AI | Healthcare AI


