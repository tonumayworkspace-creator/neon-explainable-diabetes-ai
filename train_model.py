import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib

print("Loading training data...")
train = pd.read_csv("train.csv")

TARGET = train.columns[-1]
print(f"Detected target column: {TARGET}")

X = train.drop(columns=[TARGET])
y = train[TARGET]

# =========================
# identify column types
# =========================
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("\nNumeric columns:", len(numeric_cols))
print("Categorical columns:", len(categorical_cols))

# numeric processing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# categorical processing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# full preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# model
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="auc",
    n_jobs=-1,
    tree_method="hist"
)

# full pipeline = preprocessing + model
clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# =========================
# Split
# =========================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model with full preprocessing pipeline...")

clf.fit(X_train, y_train)

# =========================
# Evaluation
# =========================
valid_probs = clf.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, valid_probs)

valid_pred = (valid_probs >= 0.5).astype(int)
acc = accuracy_score(y_valid, valid_pred)

print(f"Validation ROC-AUC: {auc:.4f}")
print(f"Validation Accuracy: {acc:.4f}")

# =========================
# Save model
# =========================
joblib.dump(clf, "diabetes_pipeline.pkl")

print("\nModel training complete.")
print("Saved file: diabetes_pipeline.pkl")
