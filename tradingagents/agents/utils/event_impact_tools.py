from pathlib import Path
from typing import Annotated, List, Dict, Any
import pandas as pd
import json
from langchain_core.tools import tool


@tool
def get_event_data_from_csv(
    ticker: Annotated[str, "Ticker symbol"],
    trade_date: Annotated[str, "Trade date in yyyy-mm-dd format (yyyy-mm-dd)"],
) -> str:
    """
    Load all discrete event rows for a given ticker from a local CSV
    and return them as JSON (list[dict]).

    Uses combined_kaggle_ectsum_100rows.csv located next to this file.
    The trade_date argument is accepted but ignored.
    """
    # Directory where THIS file lives
    base_dir = Path(__file__).resolve().parent

    # CSV is in the same folder as this file
    event_path = base_dir / "combined_kaggle_ectsum_100rows.csv"
    print(f"[DEBUG] Loading event file from: {event_path}")

    if not event_path.exists():
        return json.dumps(
            {"events": [], "error": f"File not found: {event_path}"},
            ensure_ascii=False,
        )

    df = pd.read_csv(event_path)

    # Filter ONLY by ticker (case-insensitive)
    mask = df["Ticker"].astype(str).str.upper() == str(ticker).upper()
    rows = df.loc[mask].to_dict(orient="records")

    print(f"[DEBUG] get_event_data_from_csv → ticker={ticker}, rows={len(rows)}")
    return json.dumps({"events": rows}, ensure_ascii=False)
