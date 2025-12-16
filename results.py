import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Step 0: Load datasets ---
baseline = pd.read_csv("combined_kaggle_ectsum_100_baseline_4.csv")
with_event = pd.read_csv("tradingagents_batch_output_with_event_detection.csv")
without_event = pd.read_csv("tradingagents_batch_output_without_event_detection.csv")

# --- Step 1: Normalize date formats ---
baseline["Date"] = pd.to_datetime(baseline["Date"]).dt.strftime("%Y-%m-%d")
with_event["Date"] = pd.to_datetime(with_event["Date"], dayfirst=True).dt.strftime("%Y-%m-%d")
without_event["Date"] = pd.to_datetime(without_event["Date"]).dt.strftime("%Y-%m-%d")

# --- Step 2: Merge datasets on Ticker + Date ---
baseline_subset = baseline[["Ticker", "Date", "target_decision"]]
merged_with = pd.merge(baseline_subset, with_event, on=["Ticker", "Date"], how="inner")
merged_without = pd.merge(baseline_subset, without_event, on=["Ticker", "Date"], how="inner")

# --- Step 3: Calculate overall accuracy ---
accuracy_with = (merged_with["target_decision"] == merged_with["Trader_Decision"]).mean()
accuracy_without = (merged_without["target_decision"] == merged_without["Trader_Decision"]).mean()

print(f"Accuracy WITH event detection: {accuracy_with:.2%}")
print(f"Accuracy WITHOUT event detection: {accuracy_without:.2%}")

# --- Step 4: Calculate per-ticker accuracy ---
accuracy_with_ticker = (
    merged_with["target_decision"] == merged_with["Trader_Decision"]
).groupby(merged_with["Ticker"]).mean()

accuracy_without_ticker = (
    merged_without["target_decision"] == merged_without["Trader_Decision"]
).groupby(merged_without["Ticker"]).mean()

# --- Step 5: Plot bar graph ---
x = np.arange(len(accuracy_with_ticker.index))  # positions for tickers
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, accuracy_with_ticker.values * 100, width, label="With Event Detection")
plt.bar(x + width/2, accuracy_without_ticker.values * 100, width, label="Without Event Detection")

plt.title("Per-Ticker Accuracy Comparison")
plt.xlabel("Ticker")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(x, accuracy_with_ticker.index)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()