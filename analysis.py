import pandas as pd

# Load our payment dataset
df = pd.read_csv("data/payments.csv")
# Total transactions
total_transactions = len(df)

# Successful payments
successful_payments = (df["Payment_Status"] == "Success").sum()

# Failed payments
failed_payments = (df["Payment_Status"] == "Failed").sum()

# Total transaction amount
total_revenue = df["Amount"].sum()

# Revenue involved in failed payments
failed_revenue = df.loc[
    df["Payment_Status"] == "Failed",
    "Amount"
].sum()

# Recovered failed payments
recovered_payments = (
    (df["Payment_Status"] == "Failed") &
    (df["Recovery_Outcome"] == "Recovered")
).sum()

# Recovery rate among failed payments
recovery_rate = (
    recovered_payments / failed_payments * 100
)

# Display results
print("=" * 45)
print("AI REVENUE RECOVERY - BASIC ANALYSIS")
print("=" * 45)

print(f"Total Transactions: {total_transactions:,}")
print(f"Successful Payments: {successful_payments:,}")
print(f"Failed Payments: {failed_payments:,}")
print(f"Total Transaction Value: ₹{total_revenue:,.2f}")
print(f"Failed Payment Value: ₹{failed_revenue:,.2f}")
print(f"Recovered Payments: {recovered_payments:,}")
print(f"Recovery Rate: {recovery_rate:.2f}%")

print("=" * 45)
# ----------------------------------------
# FAILURE REASON ANALYSIS
# ----------------------------------------

failed_df = df[df["Payment_Status"] == "Failed"]

print("\n")
print("=" * 50)
print("FAILURE REASON ANALYSIS")
print("=" * 50)

failure_analysis = (
    failed_df
    .groupby("Failure_Reason")
    .agg(
        Failed_Transactions=("Transaction_ID", "count"),
        Failed_Value=("Amount", "sum")
    )
    .sort_values("Failed_Transactions", ascending=False)
)

print(failure_analysis)

# ----------------------------------------
# PAYMENT METHOD ANALYSIS
# ----------------------------------------

print("\n")
print("=" * 50)
print("PAYMENT METHOD ANALYSIS")
print("=" * 50)

payment_method_analysis = (
    df.groupby("Payment_Method")
    .agg(
        Total_Transactions=("Transaction_ID", "count"),
        Failed_Transactions=(
            "Payment_Status",
            lambda x: (x == "Failed").sum()
        ),
        Total_Value=("Amount", "sum")
    )
)

payment_method_analysis["Failure_Rate"] = (
    payment_method_analysis["Failed_Transactions"]
    / payment_method_analysis["Total_Transactions"]
    * 100
)

payment_method_analysis = payment_method_analysis.sort_values(
    "Failure_Rate",
    ascending=False
)

print(payment_method_analysis)
# ----------------------------------------
# RECOVERY OUTCOME ANALYSIS
# ----------------------------------------

print("\n")
print("=" * 50)
print("RECOVERY OUTCOME ANALYSIS")
print("=" * 50)

recovery_analysis = (
    failed_df["Recovery_Outcome"]
    .value_counts()
)

print(recovery_analysis)

print("\nRecovery Percentage:")

recovery_percentage = (
    failed_df["Recovery_Outcome"]
    .value_counts(normalize=True) * 100
)

print(recovery_percentage.round(2))