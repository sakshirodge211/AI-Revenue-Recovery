import pandas as pd
import numpy as np
from faker import Faker
import random

# -----------------------------
# SETTINGS
# -----------------------------
NUM_TRANSACTIONS = 50000

fake = Faker()
np.random.seed(42)
random.seed(42)

# -----------------------------
# BASIC DATA
# -----------------------------
transaction_ids = [f"TX{100000 + i}" for i in range(NUM_TRANSACTIONS)]
customer_ids = [f"CUST{random.randint(1000, 9999)}" for _ in range(NUM_TRANSACTIONS)]

dates = pd.date_range(
    start="2025-01-01",
    end="2026-08-30",
    periods=NUM_TRANSACTIONS
)

amounts = np.round(
    np.random.lognormal(mean=6.5, sigma=1.0, size=NUM_TRANSACTIONS),
    2
)

amounts = np.clip(amounts, 50, 50000)

payment_methods = np.random.choice(
    ["UPI", "Credit Card", "Debit Card", "Net Banking", "Wallet"],
    NUM_TRANSACTIONS,
    p=[0.45, 0.20, 0.18, 0.10, 0.07]
)

payment_status = np.random.choice(
    ["Success", "Failed"],
    NUM_TRANSACTIONS,
    p=[0.82, 0.18]
)

# -----------------------------
# FAILURE REASONS
# -----------------------------
failure_reasons = []

failure_options = [
    "Insufficient Funds",
    "Bank Error",
    "Network Error",
    "Card Expired",
    "Incorrect Details",
    "Transaction Timeout",
    "Limit Exceeded"
]

for status in payment_status:
    if status == "Failed":
        failure_reasons.append(
            random.choice(failure_options)
        )
    else:
        failure_reasons.append("None")

# -----------------------------
# CUSTOMER INFORMATION
# -----------------------------
customer_age = np.random.randint(18, 60, NUM_TRANSACTIONS)

customer_type = np.random.choice(
    ["New", "Regular", "Premium"],
    NUM_TRANSACTIONS,
    p=[0.30, 0.55, 0.15]
)

previous_successful = np.random.poisson(
    lam=7,
    size=NUM_TRANSACTIONS
)

previous_failed = np.random.poisson(
    lam=2,
    size=NUM_TRANSACTIONS
)

retry_count = np.random.randint(
    0, 4,
    NUM_TRANSACTIONS
)

device_type = np.random.choice(
    ["Android", "iOS", "Web"],
    NUM_TRANSACTIONS,
    p=[0.50, 0.30, 0.20]
)

locations = np.random.choice(
    [
        "Mumbai",
        "Delhi",
        "Bangalore",
        "Hyderabad",
        "Pune",
        "Chennai",
        "Kolkata",
        "Ahmedabad",
        "Jaipur",
        "Nagpur"
    ],
    NUM_TRANSACTIONS
)

# -----------------------------
# TIME INFORMATION
# -----------------------------
hours = dates.hour

days = dates.day_name()

# -----------------------------
# RECOVERY OUTCOME
# -----------------------------
recovery_outcome = []

for i in range(NUM_TRANSACTIONS):

    if payment_status[i] == "Success":
        recovery_outcome.append("Not Required")

    else:
        # Higher chance of recovery for customers
        # with good payment history
        score = (
            previous_successful[i] * 0.08
            - previous_failed[i] * 0.05
            - retry_count[i] * 0.08
        )

        probability = 0.30 + score

        probability = max(0.05, min(0.90, probability))

        if random.random() < probability:
            recovery_outcome.append("Recovered")
        else:
            recovery_outcome.append("Not Recovered")

# -----------------------------
# CREATE DATAFRAME
# -----------------------------
data = pd.DataFrame({
    "Transaction_ID": transaction_ids,
    "Customer_ID": customer_ids,
    "Transaction_Date": dates,
    "Amount": amounts,
    "Payment_Method": payment_methods,
    "Payment_Status": payment_status,
    "Failure_Reason": failure_reasons,
    "Customer_Age": customer_age,
    "Customer_Type": customer_type,
    "Previous_Successful_Payments": previous_successful,
    "Previous_Failed_Payments": previous_failed,
    "Retry_Count": retry_count,
    "Device_Type": device_type,
    "Location": locations,
    "Hour": hours,
    "Day": days,
    "Recovery_Outcome": recovery_outcome
})

# -----------------------------
# SAVE DATASET
# -----------------------------
output_path = "data/payments.csv"
data.to_csv(
    output_path,
    index=False
)

print("====================================")
print("Dataset created successfully!")
print("====================================")
print(f"Rows: {len(data)}")
print(f"Columns: {len(data.columns)}")
print(f"Saved to: {output_path}")