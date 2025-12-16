import pandas as pd

# Load datasets
baseline = pd.read_csv("combined_kaggle_ectsum_100_baseline_4.csv")
with_event = pd.read_csv("tradingagents_batch_output_with_event_detection.csv")
without_event = pd.read_csv("tradingagents_batch_output_without_event_detection.csv")

# --- Step 1: Normalize date formats ---
# Baseline has YYYY-MM-DD, with_event has DD-MM-YYYY, without_event has YYYY-MM-DD
baseline["Date"] = pd.to_datetime(baseline["Date"]).dt.strftime("%Y-%m-%d")
with_event["Date"] = pd.to_datetime(with_event["Date"], dayfirst=True).dt.strftime("%Y-%m-%d")
without_event["Date"] = pd.to_datetime(without_event["Date"]).dt.strftime("%Y-%m-%d")

# --- Step 2: Merge datasets on Ticker + Date ---
baseline_subset = baseline[["Ticker", "Date", "target_decision"]]

merged_with = pd.merge(baseline_subset, with_event, on=["Ticker", "Date"], how="inner")
merged_without = pd.merge(baseline_subset, without_event, on=["Ticker", "Date"], how="inner")

# --- Step 3: Calculate accuracy ---
accuracy_with = (merged_with["target_decision"] == merged_with["Trader_Decision"]).mean()
accuracy_without = (merged_without["target_decision"] == merged_without["Trader_Decision"]).mean()

print(f"Accuracy WITH event detection: {accuracy_with: }")
print(f"Accuracy WITHOUT event detection: {accuracy_without: }")

# --- Optional: Show mismatches ---
mismatches_with = merged_with[merged_with["target_decision"] != merged_with["Trader_Decision"]]
mismatches_without = merged_without[merged_without["target_decision"] != merged_without["Trader_Decision"]]

print("\nSample mismatches WITH event detection:")
print(mismatches_with.head())

print("\nSample mismatches WITHOUT event detection:")
print(mismatches_without.head())