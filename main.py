from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
import pandas as pd
from dotenv import load_dotenv
import os
import time
import random
import re

# Load environment variables from .env file
load_dotenv()

def _get_retry_after_seconds(err) -> float:
    msg = str(err).lower()
    # Try to honor "Please try again in 342ms" or "in 3.2s"
    m = re.search(r"try again in\s+(\d+)\s*ms", msg)
    if m:
        return max(float(m.group(1)) / 1000.0, 0.25)
    m = re.search(r"try again in\s+([0-9]*\.?[0-9]+)\s*s", msg)
    if m:
        return max(float(m.group(1)), 0.25)
    return 0.0

def _is_rate_limit_error(err) -> bool:
    msg = str(err).lower()
    return ("rate limit" in msg) or ("rate_limit_exceeded" in msg) or ("status 429" in msg)

def call_with_retries(func, *args, max_retries=6, base_delay=0.75, max_delay=20.0, **kwargs):
    """Exponential backoff + small jitter; honors server-provided wait hints in error text."""
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries:
                suggested = _get_retry_after_seconds(e)
                wait = min(suggested if suggested > 0 else base_delay * (2 ** attempt), max_delay)
                wait += random.uniform(0.0, 0.3)  # jitter
                print(f"Rate limit hit (attempt {attempt+1}/{max_retries}). Sleeping {wait:.2f}s …")
                time.sleep(wait)
                attempt += 1
                continue
            raise

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4o-mini"  # Use a different model
config["quick_think_llm"] = "gpt-4o-mini"  # Use a different model
config["max_debate_rounds"] = 1  # Increase debate rounds

# Configure data vendors (default uses yfinance and alpha_vantage)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",           # Options: yfinance, alpha_vantage, local
    "technical_indicators": "yfinance",      # Options: yfinance, alpha_vantage, local
    "fundamental_data": "alpha_vantage",     # Options: openai, alpha_vantage, local
    "news_data": "alpha_vantage",            # Options: openai, alpha_vantage, google, local
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
'''_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)'''

df = pd.read_csv('combined_kaggle_ectsum_input.csv')
df = df[0:1]
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

output_path = "eval_results/tradingagents_batch_output_with_event_detection.csv"

if not os.path.exists(output_path):
    pd.DataFrame(columns=["Ticker", "Date", "Trader_Decision"]).to_csv(output_path, index=False)

api_counter = 0  # Track number of API calls
per_call_delay = 0.25   # small pacing between calls (seconds)
batch_sleep_every = 5   # after N successful API calls
batch_sleep_secs = 30   # sleep duration (seconds)

for i, row in df.iterrows():
    ticker = row['Ticker']
    date = row['Date'].strftime('%d-%m-%Y')

    try:
        #state, decision = ta.propagate(ticker, date)
        state, decision = call_with_retries(
            ta.propagate,
            ticker, date,                  # <-- positional args, no keywords
            max_retries=7,
            base_delay=0.75,
            max_delay=25.0
        )


        print(f"[{i+1}/{len(df)}]  {ticker} on {date} → {decision}")
        result = {
            "Ticker": ticker,
            "Date": date,
            "Trader_Decision": decision
        }
        api_counter += 1  # Increment only on successful API call
    except Exception as e:
        print(f"[{i+1}/{len(df)}] {ticker} on {date} → ERROR: {e}")
        result = {
            "Ticker": ticker,
            "Date": date,
            "Trader_Decision": "ERROR"
            #"Error": str(e)
        }

    pd.DataFrame([result]).to_csv(output_path, mode='a', header=False, index=False)

    #  Sleep after every 5 calls
    time.sleep(per_call_delay)

    #  Heavier pause every N successful calls
    if api_counter > 0 and api_counter % batch_sleep_every == 0:
        print(f" Sleeping for {batch_sleep_secs} seconds to respect API pacing...")
        time.sleep(batch_sleep_secs)
