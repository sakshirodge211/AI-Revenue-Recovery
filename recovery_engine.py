import pandas as pd

# ----------------------------------------
# 1. LOAD ML PREDICTIONS
# ----------------------------------------

df = pd.read_csv("data/recovery_predictions.csv")

# ----------------------------------------
# 2. CREATE RECOVERY PRIORITY
# ----------------------------------------

def recovery_priority(probability):
    if probability >= 75:
        return "HIGH"
    elif probability >= 50:
        return "MEDIUM"
    else:
        return "LOW"


df["Priority"] = df["Recovery_Probability"].apply(
    recovery_priority
)

# ----------------------------------------
# 3. RECOMMEND RECOVERY ACTION
# ----------------------------------------

def recovery_action(row):

    probability = row["Recovery_Probability"]
    failure = row["Failure_Reason"]

    if probability >= 75:
        return "Retry Payment"

    elif probability >= 50:

        if failure == "Insufficient Funds":
            return "Send Payment Reminder"

        elif failure in ["Network Error", "Transaction Timeout"]:
            return "Retry After Short Delay"

        elif failure == "Card Expired":
            return "Request Updated Card"

        elif failure == "Incorrect Details":
            return "Request Correct Details"

        else:
            return "Offer Alternate Payment Method"

    else:
        return "Do Not Retry Automatically"


df["Recommended_Action"] = df.apply(
    recovery_action,
    axis=1
)

# ----------------------------------------
# 4. ESTIMATE RECOVERABLE REVENUE
# ----------------------------------------

df["Expected_Recovery_Value"] = (
    df["Amount"] *
    df["Recovery_Probability"] / 100
)

# ----------------------------------------
# 5. SAVE AI DECISIONS
# ----------------------------------------

df.to_csv(
    "data/recovery_decisions.csv",
    index=False
)

# ----------------------------------------
# 6. DISPLAY RESULTS
# ----------------------------------------

print("=" * 60)
print("AI REVENUE RECOVERY DECISION ENGINE")
print("=" * 60)

print("\nPriority Distribution:")
print(df["Priority"].value_counts())

print("\nRecommended Actions:")
print(df["Recommended_Action"].value_counts())

print("\nEstimated Recoverable Revenue:")
print(
    f"₹{df['Expected_Recovery_Value'].sum():,.2f}"
)

print("\nSample AI Decisions:")
print(
    df[
        [
            "Transaction_ID",
            "Amount",
            "Failure_Reason",
            "Recovery_Probability",
            "Priority",
            "Recommended_Action",
            "Expected_Recovery_Value"
        ]
    ].head(10)
)

print("\n" + "=" * 60)
print("Recovery decisions saved successfully!")
print("File: data/recovery_decisions.csv")
print("=" * 60)
# ----------------------------------------
# 7. RANK RECOVERY OPPORTUNITIES
# ----------------------------------------

df["Recovery_Rank"] = (
    df["Expected_Recovery_Value"]
    .rank(method="first", ascending=False)
    .astype(int)
)

top_opportunities = df.sort_values(
    "Expected_Recovery_Value",
    ascending=False
).head(20)

print("\n")
print("=" * 70)
print("TOP 20 RECOVERY OPPORTUNITIES")
print("=" * 70)

print(
    top_opportunities[
        [
            "Transaction_ID",
            "Amount",
            "Failure_Reason",
            "Recovery_Probability",
            "Priority",
            "Recommended_Action",
            "Expected_Recovery_Value"
        ]
    ].to_string(index=False)
)

# Save ranked results
df.to_csv(
    "data/recovery_decisions.csv",
    index=False
)

print("\nTop recovery opportunities calculated successfully!")