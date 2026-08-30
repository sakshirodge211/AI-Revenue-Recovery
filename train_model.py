import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------------
# 1. LOAD DATA
# ----------------------------------------

df = pd.read_csv("data/payments.csv")

# We only want failed payments
df = df[df["Payment_Status"] == "Failed"].copy()

print("Failed payments used for ML:", len(df))

# ----------------------------------------
# 2. CREATE TARGET
# ----------------------------------------

df["Recovered"] = (
    df["Recovery_Outcome"] == "Recovered"
).astype(int)

# ----------------------------------------
# 3. SELECT FEATURES
# ----------------------------------------

features = [
    "Amount",
    "Payment_Method",
    "Failure_Reason",
    "Customer_Age",
    "Customer_Type",
    "Previous_Successful_Payments",
    "Previous_Failed_Payments",
    "Retry_Count",
    "Device_Type",
    "Location",
    "Hour"
]

X = df[features]
y = df["Recovered"]

# ----------------------------------------
# 4. CATEGORICAL & NUMERICAL FEATURES
# ----------------------------------------

categorical_features = [
    "Payment_Method",
    "Failure_Reason",
    "Customer_Type",
    "Device_Type",
    "Location"
]

numeric_features = [
    "Amount",
    "Customer_Age",
    "Previous_Successful_Payments",
    "Previous_Failed_Payments",
    "Retry_Count",
    "Hour"
]

# ----------------------------------------
# 5. PREPROCESSING
# ----------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)

# ----------------------------------------
# 6. CREATE MODEL
# ----------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ----------------------------------------
# 7. SPLIT DATA
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------------------
# 8. TRAIN MODEL
# ----------------------------------------

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")

# ----------------------------------------
# 9. MAKE PREDICTIONS
# ----------------------------------------

y_pred = pipeline.predict(X_test)

# ----------------------------------------
# 10. EVALUATE MODEL
# ----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n")
print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Recovered", "Recovered"]
))

print("=" * 50)
# ----------------------------------------
# 11. RECOVERY PROBABILITY
# ----------------------------------------

# Get probability of Recovery = 1
recovery_probability = pipeline.predict_proba(X_test)[:, 1]

# Create prediction results
prediction_results = X_test.copy()

prediction_results["Transaction_ID"] = df.loc[
    X_test.index,
    "Transaction_ID"
]

prediction_results["Actual_Outcome"] = y_test.map({
    0: "Not Recovered",
    1: "Recovered"
})

prediction_results["Predicted_Outcome"] = [
    "Recovered" if p >= 0.50 else "Not Recovered"
    for p in recovery_probability
]

prediction_results["Recovery_Probability"] = (
    recovery_probability * 100
).round(2)

prediction_results["Amount"] = df.loc[
    X_test.index,
    "Amount"
]

# Save predictions
prediction_results.to_csv(
    "data/recovery_predictions.csv",
    index=False
)

print("\nRecovery predictions saved successfully!")
print("File: data/recovery_predictions.csv")

print("\nSample predictions:")
print(
    prediction_results[
        [
            "Transaction_ID",
            "Amount",
            "Actual_Outcome",
            "Predicted_Outcome",
            "Recovery_Probability"
        ]
    ].head(10)
)