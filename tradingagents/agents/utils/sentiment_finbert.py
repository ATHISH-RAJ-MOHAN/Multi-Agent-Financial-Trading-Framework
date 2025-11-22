# tradingagents/agents/utils/sentiment_finbert.py
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import threading

_MODEL = {"tokenizer": None, "model": None, "lock": threading.Lock()}

def _load_model(model_id: str, device: str):
    with _MODEL["lock"]:
        if _MODEL["model"] is None:
            tok = AutoTokenizer.from_pretrained(model_id)
            mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
            device = device if device in ("cuda", "mps") and torch.cuda.is_available() else "cpu"
            mdl.to(device)
            _MODEL.update({"tokenizer": tok, "model": mdl, "device": device})
    return _MODEL["tokenizer"], _MODEL["model"], _MODEL["device"]

def score_finbert(
    texts: List[str],
    model_id: str = "yiyanghkust/finbert-tone",
    device: str = "cpu",
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Returns list of {"pos": float, "neu": float, "neg": float, "label": str}
    in the same order as texts.
    """
    print("Inside score_finbert")
    tok, mdl, dev = _load_model(model_id, device)
    mdl.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            for p in probs:
                # FinBERT tone ordering is often: [neutral, positive, negative] or similar.
                # We detect by model id; adjust if needed.
                if "yiyanghkust" in model_id.lower():
                    neu, pos, neg = p
                else:
                    # ProsusAI/finbert commonly: [positive, negative, neutral]
                    pos, neg, neu = p
                label = ["negative","neutral","positive"][ int(max(range(3), key=[neg,neu,pos].__getitem__)) ]
                out.append({"pos": float(pos), "neu": float(neu), "neg": float(neg), "label": label})
    #print(out)
    return out

# Expect a list of dicts: {"pos": float, "neu": float, "neg": float, "label": "positive|neutral|negative"}
