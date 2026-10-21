"""
Agent Tools Module (Phase 1 & Phase 2 & Phase 3)

Exposes discrete tools for the AI agent loop:
- scrape_source: Ingest data from a single channel
- check_data_quality: Audit row counts, missing values, source drops, and anomaly flags
- run_sentiment: Deterministic/classical NLP scoring (vader or transformer)
- run_topic_model: Unsupervised topic modeling (lda or bertopic) with overlap detection
- get_flagged_reviews: Drill down into negative reviews for specific themes/products
- compare_to_previous_run: Persistent SQLite memory comparison across historical runs
- draft_alert: Human-in-the-loop alert drafting (requires human approval)
- save_final_report: Finalizes processed datasets, executive summary, and run log
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from pipeline.agent_memory import (
    compare_to_previous_run as memory_compare,
    record_run,
    save_alert_draft,
)
from pipeline.insight_agent import generate_executive_summary
from pipeline.preprocessing import basic_clean, load_and_merge, preprocess_dataframe
from pipeline.sentiment_analysis import add_sentiment
from pipeline.topic_modeling import run_bertopic, run_lda

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
LOG_DIR = Path("logs")

log = logging.getLogger("agent_tools")


class AgentSessionState:
    """Encapsulates the in-memory data artifact state during an agent run."""

    def __init__(self, run_id: Optional[str] = None, dry_run: bool = False):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dry_run = dry_run
        self.scraped_sources: Dict[str, str] = {}
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.analyzed_df: Optional[pd.DataFrame] = None
        self.topic_keywords: Dict[int, List[str]] = {}
        self.sentiment_stats: Dict[str, Any] = {}
        self.drafted_alerts: List[Dict[str, Any]] = []
        self.report_saved: bool = False
        self.executive_summary: str = ""

    def get_effective_csvs(self) -> List[str]:
        """Returns CSV files available from scraping or already existing on disk."""
        if self.scraped_sources:
            return list(self.scraped_sources.values())
        # Fallback to existing disk files if dry-run or before scraping
        existing = [str(p) for p in DATA_DIR.glob("*.csv") if "checkpoint" not in p.name]
        if not existing and (DATA_DIR / "sample_reviews.csv").exists():
            existing = [str(DATA_DIR / "sample_reviews.csv")]
        return existing

    def ensure_dataframe_loaded(self) -> pd.DataFrame:
        """Loads and preprocessed dataframe if not already prepared."""
        if self.cleaned_df is not None:
            return self.cleaned_df
        csv_paths = self.get_effective_csvs()
        if not csv_paths:
            raise RuntimeError("No data files available. Run scrape_source or check data/ folder.")
        merged = load_and_merge(csv_paths)
        cleaned = preprocess_dataframe(merged)
        self.raw_df = merged
        self.cleaned_df = cleaned
        return cleaned


# Source script mapping
SOURCE_SCRIPT_MAP = {
    "playstore": ("scrapers/playstore_scraper.py", DATA_DIR / "playstore_reviews.csv"),
    "amazon": ("scrapers/ecommerce_scraper.py", DATA_DIR / "amazon_reviews.csv"),
    "flipkart": ("scrapers/ecommerce_scraper.py", DATA_DIR / "flipkart_reviews.csv"),
    "website": ("scrapers/website_scraper.py", DATA_DIR / "website_reviews.csv"),
    "twitter": ("scrapers/social_scraper.py", DATA_DIR / "twitter_mentions.csv"),
    "instagram": ("scrapers/social_scraper.py", DATA_DIR / "instagram_mentions.csv"),
    "reddit": ("scrapers/social_scraper.py", DATA_DIR / "reddit_mentions.csv"),
}


# --------------------------------------------------------------------------- #
# Tool Implementations
# --------------------------------------------------------------------------- #

def tool_scrape_source(state: AgentSessionState, source: str) -> Dict[str, Any]:
    """Scrapes data for a specific source or verifies existing source data in dry-run mode."""
    source_lower = source.strip().lower()
    if source_lower not in SOURCE_SCRIPT_MAP:
        return {
            "status": "error",
            "message": f"Unsupported source '{source}'. Supported: {list(SOURCE_SCRIPT_MAP.keys())}",
        }

    script, expected_output = SOURCE_SCRIPT_MAP[source_lower]

    if state.dry_run:
        # In dry run mode, check if the file already exists or fallback to sample
        if expected_output.exists():
            row_count = len(pd.read_csv(expected_output))
            state.scraped_sources[source_lower] = str(expected_output)
            return {
                "status": "success",
                "mode": "dry_run",
                "source": source_lower,
                "file_path": str(expected_output),
                "row_count": row_count,
                "message": f"Source '{source_lower}' verified with {row_count} rows from disk.",
            }
        sample_path = DATA_DIR / "sample_reviews.csv"
        if sample_path.exists():
            state.scraped_sources[source_lower] = str(sample_path)
            return {
                "status": "success",
                "mode": "dry_run",
                "source": source_lower,
                "file_path": str(sample_path),
                "row_count": len(pd.read_csv(sample_path)),
                "message": f"Source '{source_lower}' using bundled sample data ({sample_path}).",
            }
        return {"status": "warning", "message": f"No existing data file for source '{source_lower}'."}

    # Execute actual scraper
    try:
        res = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if res.returncode != 0:
            return {
                "status": "failure",
                "source": source_lower,
                "error": res.stderr[-500:] if res.stderr else "Unknown scraper execution error",
                "suggestion": "Check network connectivity or credentials; fallback data on disk may be used.",
            }

        if expected_output.exists():
            df_source = pd.read_csv(expected_output)
            row_count = len(df_source)
            state.scraped_sources[source_lower] = str(expected_output)
            # Invalidate cached df so new data is incorporated
            state.cleaned_df = None
            return {
                "status": "success",
                "source": source_lower,
                "file_path": str(expected_output),
                "row_count": row_count,
            }
        else:
            return {
                "status": "failure",
                "source": source_lower,
                "error": f"Scraper finished but expected output file {expected_output} was not created.",
            }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "source": source_lower, "error": "Scraper timed out after 15m"}
    except Exception as e:
        return {"status": "error", "source": source_lower, "error": str(e)}


def tool_check_data_quality(state: AgentSessionState) -> Dict[str, Any]:
    """Audits row counts per source, missing text, duplicate rates, and flags anomalies."""
    csv_paths = state.get_effective_csvs()
    if not csv_paths:
        return {
            "status": "anomaly",
            "flags": ["NO_DATA_AVAILABLE"],
            "total_files": 0,
            "message": "No CSV files discovered in data/ or scraped.",
        }

    source_counts = {}
    total_raw_rows = 0
    total_empty_rows = 0
    flags = []

    for path in csv_paths:
        try:
            df_item = pd.read_csv(path)
            cnt = len(df_item)
            src_name = Path(path).stem.replace("_reviews", "").replace("_mentions", "")
            source_counts[src_name] = cnt
            total_raw_rows += cnt

            if "review_text" in df_item.columns:
                empty_cnt = df_item["review_text"].isna().sum() + (df_item["review_text"].astype(str).str.strip() == "").sum()
                total_empty_rows += empty_cnt

            # Anomaly check: source exists but is completely empty
            if cnt == 0:
                flags.append(f"EMPTY_SOURCE: '{src_name}' returned 0 rows.")
            elif cnt < 15:
                flags.append(f"LOW_DATA_VOLUME: '{src_name}' has only {cnt} rows.")
        except Exception as e:
            flags.append(f"CORRUPT_FILE: Could not read {path}: {e}")

    # Load and clean to evaluate clean row count
    try:
        cleaned_df = state.ensure_dataframe_loaded()
        cleaned_rows = len(cleaned_df)
    except Exception as e:
        return {"status": "error", "message": f"Failed to merge and clean data: {e}"}

    empty_rate = round((total_empty_rows / total_raw_rows) * 100, 2) if total_raw_rows > 0 else 0.0
    if empty_rate > 25.0:
        flags.append(f"HIGH_EMPTY_RATE: {empty_rate}% of reviews were empty/null.")

    if cleaned_rows < 10:
        flags.append("INSUFFICIENT_CORPUS: Fewer than 10 valid reviews after cleaning.")

    return {
        "status": "anomalies_detected" if flags else "healthy",
        "total_files_inspected": len(csv_paths),
        "total_raw_reviews": total_raw_rows,
        "clean_reviews_ready_for_nlp": cleaned_rows,
        "counts_by_source": source_counts,
        "empty_or_null_percentage": empty_rate,
        "flags": flags,
        "recommendation": (
            "Proceed with sentiment scoring"
            if cleaned_rows >= 10
            else "Attempt re-scraping missing sources or inspect data files"
        ),
    }


def tool_run_sentiment(state: AgentSessionState, method: str = "vader") -> Dict[str, Any]:
    """Scores review sentiment using VADER or Transformer, updating session state."""
    df = state.ensure_dataframe_loaded()

    method_clean = method.lower().strip()
    if method_clean not in ["vader", "transformer"]:
        return {"status": "error", "message": f"Invalid method '{method}'. Choose 'vader' or 'transformer'."}

    df_scored = add_sentiment(df, method=method_clean)
    state.analyzed_df = df_scored

    counts = df_scored["sentiment_label"].value_counts().to_dict()
    total = len(df_scored)
    pct_pos = round((counts.get("positive", 0) / total) * 100, 2) if total else 0.0
    pct_neg = round((counts.get("negative", 0) / total) * 100, 2) if total else 0.0
    pct_neu = round((counts.get("neutral", 0) / total) * 100, 2) if total else 0.0

    state.sentiment_stats = {
        "method": method_clean,
        "total_scored": total,
        "counts": counts,
        "pct_positive": pct_pos,
        "pct_negative": pct_neg,
        "pct_neutral": pct_neu,
        "mean_compound": round(float(df_scored["sentiment_score"].mean()), 3),
    }

    return {
        "status": "success",
        "method": method_clean,
        "total_scored": total,
        "pct_positive": pct_pos,
        "pct_negative": pct_neg,
        "pct_neutral": pct_neu,
        "mean_compound_score": state.sentiment_stats["mean_compound"],
    }


def tool_run_topic_model(
    state: AgentSessionState, method: str = "lda", n_topics: int = 6
) -> Dict[str, Any]:
    """
    Extracts topics/themes using LDA or BERTopic.
    Evaluates topic separation and detects if topics form a single overlapping blob.
    """
    if state.analyzed_df is not None:
        df = state.analyzed_df
    else:
        df = state.ensure_dataframe_loaded()
        if "sentiment_label" not in df.columns:
            df = add_sentiment(df, method="vader")
            state.analyzed_df = df

    method_clean = method.lower().strip()
    # Sensible bounds for small vs large datasets
    if len(df) < 50:
        min_df_val = 1
        max_df_val = 1.0
    elif len(df) < 200:
        min_df_val = 2
        max_df_val = 0.95
    else:
        min_df_val = 5
        max_df_val = 0.9

    overlap_detected = False
    blob_flag = False

    if method_clean == "lda":
        topic_keywords, df_topics = run_lda(
            df, n_topics=n_topics, min_df=min_df_val, max_df=max_df_val
        )
        state.topic_keywords = topic_keywords
        state.analyzed_df = df_topics

        # Check for topic overlap (giant blob heuristic):
        # Calculate Jaccard similarity between top keywords of different topics
        all_word_sets = [set(words[:5]) for words in topic_keywords.values()]
        shared_pairs = 0
        total_pairs = 0
        for i in range(len(all_word_sets)):
            for j in range(i + 1, len(all_word_sets)):
                total_pairs += 1
                intersection = len(all_word_sets[i].intersection(all_word_sets[j]))
                if intersection >= 2:
                    shared_pairs += 1

        if total_pairs > 0 and (shared_pairs / total_pairs) >= 0.3:
            overlap_detected = True
            blob_flag = True

    elif method_clean == "bertopic":
        try:
            _, df_topics = run_bertopic(df)
            state.analyzed_df = df_topics
            topic_keywords = {}
            for t_idx, grp in df_topics.groupby("dominant_topic"):
                if t_idx != -1:
                    topic_keywords[int(t_idx)] = grp["topic_label"].iloc[0].split(" / ")[:5]
            state.topic_keywords = topic_keywords
        except Exception as e:
            return {"status": "error", "message": f"BERTopic execution failed: {e}. Try method='lda'."}
    else:
        return {"status": "error", "message": "Method must be 'lda' or 'bertopic'."}

    # Format human readable summary of topics
    topic_summary = {}
    for idx, words in state.topic_keywords.items():
        topic_summary[f"Topic_{idx}"] = words[:5]

    return {
        "status": "success",
        "method": method_clean,
        "n_topics_extracted": len(state.topic_keywords),
        "topics": topic_summary,
        "overlap_detected": overlap_detected,
        "recommendation": (
            "Topics are highly overlapping (single blob). Consider re-running with n_topics=4 or method='bertopic'."
            if blob_flag
            else "Topic separation is distinct and coherent."
        ),
    }


def tool_get_flagged_reviews(
    state: AgentSessionState, theme: Optional[str] = None, n: int = 5
) -> Dict[str, Any]:
    """Retrieves the most severe negative reviews, optionally filtered by keyword/theme."""
    if state.analyzed_df is None:
        return {"status": "error", "message": "Must run sentiment scoring before querying flagged reviews."}

    df = state.analyzed_df
    neg_df = df[df["sentiment_label"] == "negative"]

    if theme and theme.strip():
        term = theme.lower().strip()
        text_match = neg_df["review_text"].str.lower().str.contains(term, na=False)
        if "topic_label" in neg_df.columns:
            theme_match = neg_df["topic_label"].astype(str).str.lower().str.contains(term, na=False)
            matched = neg_df[text_match | theme_match]
        else:
            matched = neg_df[text_match]
        if len(matched) > 0:
            neg_df = matched

    sorted_neg = neg_df.sort_values("sentiment_score", ascending=True).head(n)
    samples = []
    for _, row in sorted_neg.iterrows():
        samples.append({
            "product": row.get("product", "Unknown"),
            "source": row.get("source", "Unknown"),
            "sentiment_score": round(float(row.get("sentiment_score", -1.0)), 3),
            "theme": row.get("topic_label", "General"),
            "text": str(row.get("review_text", ""))[:280],
        })

    return {
        "status": "success",
        "filter_theme": theme,
        "flagged_count": len(samples),
        "flagged_reviews": samples,
    }


def tool_compare_to_previous_run(state: AgentSessionState) -> Dict[str, Any]:
    """Compares current run's metrics to previous runs stored in runs.db SQLite memory."""
    curr_n = len(state.analyzed_df) if state.analyzed_df is not None else 0
    pct_pos = state.sentiment_stats.get("pct_positive", 0.0)
    pct_neg = state.sentiment_stats.get("pct_negative", 0.0)

    # Top theme counts
    theme_counts = {}
    if state.analyzed_df is not None and "topic_label" in state.analyzed_df.columns:
        theme_counts = state.analyzed_df["topic_label"].value_counts().head(5).to_dict()

    current_metrics = {
        "n_reviews": curr_n,
        "pct_positive": pct_pos,
        "pct_negative": pct_neg,
        "top_themes": theme_counts,
    }

    comparison = memory_compare(current_metrics)
    return {
        "status": "success",
        "comparison": comparison,
    }


def tool_draft_alert(
    state: AgentSessionState, severity: str, reason: str, metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Drafts an escalation alert requiring human checkpoint approval.
    Writes to outputs/pending_alerts.json. Does NOT auto-send.
    """
    alert_id = f"ALT-{state.run_id}-{len(state.drafted_alerts) + 1:02d}"
    alert_record = save_alert_draft(
        alert_id=alert_id,
        run_id=state.run_id,
        severity=severity,
        reason=reason,
        metrics=metrics or state.sentiment_stats,
    )
    state.drafted_alerts.append(alert_record)

    return {
        "status": "drafted_pending_human_approval",
        "alert_id": alert_id,
        "severity": severity.upper(),
        "reason": reason,
        "human_in_the_loop_message": (
            f"Alert {alert_id} drafted and queued. A human administrator must review and approve "
            f"it before dispatch. View outputs/pending_alerts.json for details."
        ),
    }


def tool_save_final_report(state: AgentSessionState) -> Dict[str, Any]:
    """Saves processed CSVs, generates the executive summary, logs to SQLite runs.db, and finalizes run."""
    if state.analyzed_df is None:
        return {"status": "error", "message": "Cannot save final report without analyzed dataset."}

    df = state.analyzed_df
    run_id = state.run_id
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save CSV artifacts (timestamped + latest for dashboard compatibility)
    processed_path = OUTPUT_DIR / f"processed_reviews_{run_id}.csv"
    latest_path = OUTPUT_DIR / "processed_reviews_latest.csv"
    df.to_csv(processed_path, index=False)
    df.to_csv(latest_path, index=False)

    # 2. Generate and save executive summary
    summary_text = generate_executive_summary(df, backend="gemini")
    state.executive_summary = summary_text

    summary_path = OUTPUT_DIR / f"executive_summary_{run_id}.txt"
    latest_summary_path = OUTPUT_DIR / "executive_summary_latest.txt"
    summary_path.write_text(summary_text)
    latest_summary_path.write_text(summary_text)

    # 3. Record in runs.db SQLite memory
    pos = state.sentiment_stats.get("pct_positive", 0.0)
    neg = state.sentiment_stats.get("pct_negative", 0.0)
    neu = state.sentiment_stats.get("pct_neutral", 0.0)

    top_themes = {}
    if "topic_label" in df.columns:
        top_themes = df["topic_label"].value_counts().head(6).to_dict()

    source_counts = {}
    if "source" in df.columns:
        source_counts = df["source"].value_counts().to_dict()

    record_run(
        run_id=run_id,
        n_reviews=len(df),
        pct_positive=pos,
        pct_negative=neg,
        pct_neutral=neu,
        top_themes=top_themes,
        source_breakdown=source_counts,
        alerts_count=len(state.drafted_alerts),
        metadata={"dry_run": state.dry_run},
    )

    state.report_saved = True

    return {
        "status": "success",
        "run_id": run_id,
        "processed_rows": len(df),
        "artifacts_saved": [
            str(processed_path),
            str(latest_path),
            str(summary_path),
            str(latest_summary_path),
        ],
        "database_logged": "data/runs.db",
        "alerts_count": len(state.drafted_alerts),
    }


# --------------------------------------------------------------------------- #
# Tool Schemas for LLM (Anthropic & Gemini standard tool format)
# --------------------------------------------------------------------------- #

AGENT_TOOLS_SCHEMA = [
    {
        "name": "scrape_source",
        "description": "Scrapes review/mention data from an external channel (playstore, amazon, flipkart, website, twitter, instagram, reddit).",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The data source to scrape: 'playstore', 'amazon', 'flipkart', 'website', 'twitter', 'instagram', or 'reddit'.",
                    "enum": ["playstore", "amazon", "flipkart", "website", "twitter", "instagram", "reddit"],
                }
            },
            "required": ["source"],
        },
    },
    {
        "name": "check_data_quality",
        "description": "Performs a comprehensive audit of ingested datasets: row counts per source, empty review rate, missing files, and anomaly flags.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "run_sentiment",
        "description": "Scores review sentiment using classical/discriminative NLP (VADER or DistilBERT transformer). Returns sentiment breakdown percentages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "Algorithm to use: 'vader' (fast, robust for slang/emoji) or 'transformer' (slower, nuanced).",
                    "enum": ["vader", "transformer"],
                    "default": "vader",
                }
            },
        },
    },
    {
        "name": "run_topic_model",
        "description": "Discovers latent themes/topics across customer feedback using LDA or BERTopic. Automatically detects topic overlap and single-blob issues.",
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "Topic extraction method: 'lda' (fast, bag of words) or 'bertopic' (embeddings-based).",
                    "enum": ["lda", "bertopic"],
                    "default": "lda",
                },
                "n_topics": {
                    "type": "integer",
                    "description": "Number of topics to fit (default: 6).",
                    "default": 6,
                },
            },
        },
    },
    {
        "name": "get_flagged_reviews",
        "description": "Retrieves the most severe negative customer reviews, optionally filtered by theme or hardware keyword (e.g., 'battery', 'connectivity').",
        "input_schema": {
            "type": "object",
            "properties": {
                "theme": {
                    "type": "string",
                    "description": "Optional keyword or theme to filter reviews (e.g. 'connectivity', 'mic', 'battery').",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of sample reviews to return (default: 5).",
                    "default": 5,
                },
            },
        },
    },
    {
        "name": "compare_to_previous_run",
        "description": "Compares current metrics against persistent SQLite run history in runs.db. Detects sudden negative sentiment spikes and volume drop anomalies.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "draft_alert",
        "description": "Drafts an escalation alert for human approval. Does NOT automatically notify external channels until a human approves it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "severity": {
                    "type": "string",
                    "description": "Alert severity level: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'.",
                    "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                },
                "reason": {
                    "type": "string",
                    "description": "Detailed explanation of the anomaly or spike justifying this escalation.",
                },
            },
            "required": ["severity", "reason"],
        },
    },
    {
        "name": "save_final_report",
        "description": "Finalizes the session: outputs processed CSVs, generates the GenAI executive summary, and logs run stats to SQLite database.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


def execute_tool(tool_name: str, tool_input: Dict[str, Any], state: AgentSessionState) -> Dict[str, Any]:
    """Dispatches tool call to appropriate Python function and returns structured result."""
    log.info(f"[Agent Tool Call] -> {tool_name} with params: {tool_input}")
    tool_input = tool_input or {}

    if tool_name == "scrape_source":
        return tool_scrape_source(state, source=tool_input.get("source", "playstore"))
    elif tool_name == "check_data_quality":
        return tool_check_data_quality(state)
    elif tool_name == "run_sentiment":
        return tool_run_sentiment(state, method=tool_input.get("method", "vader"))
    elif tool_name == "run_topic_model":
        return tool_run_topic_model(
            state,
            method=tool_input.get("method", "lda"),
            n_topics=tool_input.get("n_topics", 6),
        )
    elif tool_name == "get_flagged_reviews":
        return tool_get_flagged_reviews(
            state,
            theme=tool_input.get("theme"),
            n=tool_input.get("n", 5),
        )
    elif tool_name == "compare_to_previous_run":
        return tool_compare_to_previous_run(state)
    elif tool_name == "draft_alert":
        return tool_draft_alert(
            state,
            severity=tool_input.get("severity", "MEDIUM"),
            reason=tool_input.get("reason", "Anomaly detected in metrics"),
        )
    elif tool_name == "save_final_report":
        return tool_save_final_report(state)
    else:
        return {"status": "error", "message": f"Unknown tool name '{tool_name}'"}
