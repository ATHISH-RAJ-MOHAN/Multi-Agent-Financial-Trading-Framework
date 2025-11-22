# tradingagents/agents/analysts/event_impact_analyst.py
from langchain_core.messages import AIMessage

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage

from tradingagents.agents.utils.agent_utils import get_event_data_from_csv


ROW_SCORING_SYSTEM_MESSAGE = """
You are a financial event-impact scoring assistant.

You receive a single structured company event (row) with fields like date,
ticker, OHLCV, dividends, splits, year, and a text summary.

Your job is to assign the expected short-term (1–3 day) impact on the stock
price using this scale:

    -5  → extremely negative impact  
    -3  → clearly negative  
     0  → neutral / no meaningful impact  
    +3  → clearly positive  
    +5  → extremely positive impact  

Rules:
- Carefully use ALL provided fields, not just the summary.
- Output ONLY a JSON object with a single key "impact_score"
  whose value is a float between -5 and 5, inclusive.
- No explanation, no prose, no markdown, no commentary.
Example:
{"impact_score": -2.5}
""".strip()


def _score_single_event(llm, row: pd.Series) -> float:
    """
    Call the LLM once for a single event row and return an impact score.
    """

    # Build a structured text representation of the row
    row_lines = []
    for col in row.index:
        row_lines.append(f"{col}: {row[col]}")
    row_text = "\n".join(row_lines)

    messages = [
        SystemMessage(content=ROW_SCORING_SYSTEM_MESSAGE),
        HumanMessage(
            content=(
                f"Here is one event row for scoring:\n\n{row_text}\n\n"
                'Return ONLY JSON like {"impact_score": <float in [-5,5]>}.'
            )
        ),
    ]

    response = llm.invoke(messages)
    text = getattr(response, "content", response)

    # Defensive JSON parsing
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return 0.0
        obj = json.loads(text[start:end])
        score = float(obj.get("impact_score", 0.0))
    except Exception:
        score = 0.0

    # Clamp score to [-5, 5]
    score = max(-5.0, min(5.0, score))
    return score


def _get_events_for_state(state: Dict[str, Any]) -> pd.DataFrame:
    """
    Use the get_event_data_from_csv tool ONCE to fetch events for the
    current ticker and trade_date, then convert to a DataFrame.

    Expects:
        - state["company_of_interest"]
        - state["trade_date"] (yyyy-mm-dd string or date-like)
    """
    ticker = state.get("company_of_interest") or state.get("ticker")
    trade_date = state.get("trade_date") or state.get("date")

    if ticker is None or trade_date is None:
        raise ValueError(
            "EventImpactAnalyst: missing 'company_of_interest' / 'ticker' "
            "or 'trade_date' / 'date' in state."
        )

    if not isinstance(trade_date, str):
        trade_date = str(trade_date)

    # Call the tool via .invoke → returns the raw string
    raw_json = get_event_data_from_csv.invoke(
        {"ticker": ticker, "trade_date": trade_date}
    )

    try:
        payload = json.loads(raw_json)
    except Exception:
        payload = {"events": []}

    rows = payload.get("events", [])
    df = pd.DataFrame(rows)
    return df


def create_event_impact_analyst(llm):
    def event_impact_node(state: Dict[str, Any]) -> Dict[str, Any]:
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        # 1) Fetch events via the tool ONCE
        df = _get_events_for_state(state)
        print("unnati in create_event_impact_analyst df=", df.head())
        if df.empty:
            msg = f"No valid event data found for {ticker} on {current_date}."
            state["event_data_with_impact"] = []
            state["event_impact_report"] = msg
            return {
                "messages": [AIMessage(content=report)],
                "event_data_with_impact": [],
                "event_impact_report": msg,
            }

        # 2) Score each event row with the LLM (no tools)
        impact_scores: List[float] = []
        for _, row in df.iterrows():
            score = _score_single_event(llm, row)
            impact_scores.append(score)
            try:
                print(
                    f"[EventImpactAnalyst] Ticker={row.get('Ticker', ticker)}, "
                    f"Date={row.get('Date', current_date)}, impact_score={score:.2f}"
                )
            except Exception:
                pass

        df = df.copy()
        df["impact_score"] = impact_scores
        print("df[impact_score]: ", df["impact_score"])

        # 3) Write impact scores to a CSV so you can inspect them
        try:
            summary_col = None
            if "summary" in df.columns:
                summary_col = "summary"
            elif "Summary" in df.columns:
                summary_col = "Summary"

            out_df = df[["impact_score"]].copy()
            if summary_col is not None:
                out_df.insert(0, "summary", df[summary_col])

            out_path = Path("event_impact_scores.csv")
            out_df.to_csv(out_path, index=False, encoding="utf-8")
        except Exception as e:
            print(f"[EventImpactAnalyst] Could not write event_impact_scores.csv: {e}")

        # Filter out neutral/no-impact events
        nonzero_df = df[df["impact_score"] != 0].copy()

        if nonzero_df.empty:
            avg_score = 0.0
            pos_count = neg_count = 0
            top_events = []
        else:
            avg_score = float(nonzero_df["impact_score"].mean())
            pos_count = int((nonzero_df["impact_score"] > 0).sum())
            neg_count = int((nonzero_df["impact_score"] < 0).sum())

            # Sort by absolute strength and take top 3 impactful events for readability
            top_events_df = nonzero_df.loc[
                nonzero_df["impact_score"].abs().sort_values(ascending=False).index
            ]

            top_events = [
                f"- {row.get('Summary', '')} (impact: {row['impact_score']:.2f})"
                for _, row in top_events_df.iterrows()
            ]

            
        report = (
            f"Event Impact Analyst Report\n"
            f"- Ticker: {ticker}\n"
            f"- Date: {current_date}\n"
            f"- Total events: {len(df)}\n"
            f"- Non-neutral impactful events: {len(nonzero_df)}\n"
            f"- Positive events: {pos_count}\n"
            f"- Negative events: {neg_count}\n"
            f"- Average non-zero impact score: {avg_score:.2f}\n\n"
            f"Top impactful events:\n" + "\n".join(top_events if top_events else ["None"])
        )


        return {
            "messages": [AIMessage(content=report)],
            "event_data_with_impact": df.to_dict(orient="records"),
            "event_impact_report": report,
        }

    return event_impact_node
