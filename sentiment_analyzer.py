"""
Sentiment Analysis Engine — powered by NLTK VADER
Analyzes CSV files and returns sentiment distribution as JSON.
"""

import sys
import csv
import json
import os

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ─────────────────────────────────────────────
# Ensure VADER lexicon is available
# ─────────────────────────────────────────────
try:
    _sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
    _sia = SentimentIntensityAnalyzer()


# ─────────────────────────────────────────────
# VADER-based classifier
# ─────────────────────────────────────────────

def classify_sentiment(text: str) -> str:
    """
    Classify sentiment using NLTK VADER.
    Returns 'positive', 'negative', or 'neutral'.

    VADER compound score thresholds (industry standard):
      >= 0.05  → positive
      <= -0.05 → negative
      else     → neutral
    """
    if not text or not text.strip():
        return "neutral"

    scores = _sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"


# ─────────────────────────────────────────────
# CSV helpers (unchanged interface)
# ─────────────────────────────────────────────

def _resolve_column(fieldnames: list, text_column: str) -> str:
    """Return the actual column name to use, with fallback logic."""
    if text_column in fieldnames:
        return text_column
    # Case-insensitive match
    match = next(
        (fn for fn in fieldnames if fn.strip().lower() == text_column.lower()),
        None,
    )
    if match:
        return match
    if fieldnames:
        return fieldnames[0]
    raise ValueError("CSV has no columns.")


def analyze_csv(filepath: str, text_column: str = "text") -> dict:
    """
    Read a CSV file, classify each row's sentiment, and return distribution.
    """
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    total = 0

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        text_column = _resolve_column(fieldnames, text_column)

        for row in reader:
            raw = row.get(text_column, "")
            text = str(raw).strip() if raw is not None else ""
            sentiment = classify_sentiment(text)
            counts[sentiment] += 1
            total += 1

    if total == 0:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    return {
        "positive": round((counts["positive"] / total) * 100, 2),
        "negative": round((counts["negative"] / total) * 100, 2),
        "neutral":  round((counts["neutral"]  / total) * 100, 2),
    }


def analyze_csv_detailed(filepath: str, text_column: str = "text") -> dict:
    """
    Same as analyze_csv but also returns per-row sentiment labels and
    the raw VADER compound score for each row.
    Used by the web server.
    """
    rows_out = []
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    total = 0

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        text_column = _resolve_column(fieldnames, text_column)

        for i, row in enumerate(reader):
            raw = row.get(text_column, "")
            text = str(raw).strip() if raw is not None else ""

            # Get full VADER scores for richer output
            scores = _sia.polarity_scores(text) if text else {"compound": 0.0}
            compound = scores.get("compound", 0.0)

            if compound >= 0.05:
                sentiment = "positive"
            elif compound <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            counts[sentiment] += 1
            total += 1
            rows_out.append({
                "index":    i + 1,
                "text":     text[:120],
                "sentiment": sentiment,
                "score":    round(compound, 4),   # expose VADER compound score
            })

    if total == 0:
        distribution = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    else:
        distribution = {
            "positive": round((counts["positive"] / total) * 100, 2),
            "negative": round((counts["negative"] / total) * 100, 2),
            "neutral":  round((counts["neutral"]  / total) * 100, 2),
        }

    return {
        "distribution": distribution,
        "counts":       counts,
        "total":        total,
        "rows":         rows_out,
        "text_column":  text_column,
    }


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sentiment_analyzer.py <csv_file> [text_column]")
        sys.exit(1)

    csv_path = sys.argv[1]
    col = sys.argv[2] if len(sys.argv) > 2 else "text"

    try:
        result = analyze_csv(csv_path, col)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
