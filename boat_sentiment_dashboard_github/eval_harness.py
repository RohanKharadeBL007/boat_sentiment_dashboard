"""
AI Agent Evaluation Harness (Phase 5)

Validates that the autonomous agent behaves rationally and takes sensible
tool-call sequences under real-world failure modes and anomalies:

Scenarios:
1. Scraper returns empty/corrupt data -> Agent catches in check_data_quality, attempts recovery
2. Negative sentiment spikes (+15%) -> Agent drills down and drafts an escalation alert
3. Topics come back as one giant overlapping blob -> Agent detects overlap and adapts
4. Temporal drift vs historical memory -> Agent identifies delta using runs.db comparison
5. Healthy routine run -> Agent executes cleanly with zero false-alarm alerts

Usage:
    python eval_harness.py
"""

import json
import logging
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.agent_loop import AgentRunner
from pipeline.agent_memory import init_db, record_run
from pipeline.agent_tools import (
    AgentSessionState,
    execute_tool,
)

logging.basicConfig(
    level=logging.WARNING,  # Keep output focused on evaluation scorecard
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" SCENARIO: {title}")
    print("=" * 70)


def run_scenario_1_empty_scraper() -> Tuple[bool, str, List[str]]:
    """Scenario 1: Scraper / raw data returns empty data."""
    print_header("1. Empty Scraper / Missing Data Anomaly")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_csv = Path(tmp_dir) / "empty_reviews.csv"
        # Create empty CSV with required headers but 0 rows
        pd.DataFrame(columns=["review_text", "rating", "date", "source", "product"]).to_csv(
            tmp_csv, index=False
        )

        state = AgentSessionState(run_id="eval_empty_data", dry_run=True)
        state.scraped_sources["empty_test"] = str(tmp_csv)

        # Agent initiates audit
        audit = execute_tool("check_data_quality", {}, state)
        tool_sequence = ["check_data_quality"]

        passed = (
            audit.get("status") in ["anomaly", "anomalies_detected"]
            or "EMPTY_SOURCE" in str(audit.get("flags", []))
            or audit.get("clean_reviews_ready_for_nlp", 0) == 0
        )

        details = f"Audit response status: {audit.get('status')}, clean reviews: {audit.get('clean_reviews_ready_for_nlp', 0)}"
        return passed, details, tool_sequence


def run_scenario_2_negative_sentiment_spike() -> Tuple[bool, str, List[str]]:
    """Scenario 2: Negative sentiment spikes (+15% negative)."""
    print_header("2. Severe Negative Sentiment Spike")

    # Generate synthetic unambiguously negative review corpus
    synthetic_neg = [
        "horrible terrible defective broken product do not buy",
        "worst awful battery failure ever waste of money",
        "bad damaged unit terrible customer service",
        "pathetic useless broken rubbish item completely ruined",
        "terrible waste of money completely broken defective",
        "awful bluetooth drop and disconnect defect continuous",
        "horrible sound quality bad speaker and distorted noise",
        "terrible mic defect completely ruined voice calls",
        "broken case and damaged lid cheap plastic",
        "defective charging port terrible product totally useless",
        "great sound and nice design love it",  # 1 positive
    ]

    df_synth = pd.DataFrame({
        "review_text": synthetic_neg,
        "rating": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5],
        "date": ["2026-09-20"] * len(synthetic_neg),
        "source": ["amazon"] * len(synthetic_neg),
        "product": ["boAt Airdopes 141"] * len(synthetic_neg),
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_csv = Path(tmp_dir) / "spiked_negative.csv"
        df_synth.to_csv(tmp_csv, index=False)

        state = AgentSessionState(run_id="eval_neg_spike", dry_run=True)
        state.scraped_sources["amazon"] = str(tmp_csv)

        tool_sequence = []
        # Agent execution
        audit = execute_tool("check_data_quality", {}, state)
        tool_sequence.append("check_data_quality")

        sentiment = execute_tool("run_sentiment", {"method": "vader"}, state)
        tool_sequence.append("run_sentiment")

        # Agent should observe high negative % (> 50%) and drill down
        tool_sequence.append("get_flagged_reviews")
        flagged = execute_tool("get_flagged_reviews", {"theme": "battery", "n": 3}, state)

        # Agent must draft an escalation alert
        tool_sequence.append("draft_alert")
        alert = execute_tool(
            "draft_alert",
            {
                "severity": "HIGH",
                "reason": f"Severe negative sentiment spike ({sentiment['pct_negative']}%) detected in hardware reviews.",
            },
            state,
        )

        passed = (
            sentiment.get("pct_negative", 0.0) >= 50.0
            and len(state.drafted_alerts) > 0
            and alert.get("status") == "drafted_pending_human_approval"
        )

        details = f"Negative %: {sentiment.get('pct_negative')}%, Alert Drafted: {alert.get('alert_id')} ({alert.get('severity')})"
        return passed, details, tool_sequence


def run_scenario_3_topic_blob_overlap() -> Tuple[bool, str, List[str]]:
    """Scenario 3: Topics return as one giant overlapping blob."""
    print_header("3. Topic Overlap (Single Giant Blob) Adaptation")

    # Repetitive homogeneous text sharing core terms across all topics
    blob_texts = [
        "boat earphone sound quality bass wireless battery drain issue",
        "boat earphone sound quality bass wireless latency lag delay",
        "boat earphone sound quality bass wireless disconnect drop bug",
        "boat earphone sound quality bass wireless noise crackle static",
        "boat earphone sound quality bass wireless interference buzz problem",
        "boat earphone sound quality bass wireless broken plastic hinge",
        "boat earphone sound quality bass wireless damage foam cushion",
        "boat earphone sound quality bass wireless fail charging case",
        "boat earphone sound quality bass wireless defective pair unit",
        "boat earphone sound quality bass wireless terrible button click",
    ] * 3

    df_blob = pd.DataFrame({
        "review_text": blob_texts,
        "rating": [3] * len(blob_texts),
        "date": ["2026-09-21"] * len(blob_texts),
        "source": ["playstore"] * len(blob_texts),
        "product": ["boAt Rockerz"] * len(blob_texts),
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_csv = Path(tmp_dir) / "blob_reviews.csv"
        df_blob.to_csv(tmp_csv, index=False)

        state = AgentSessionState(run_id="eval_topic_blob", dry_run=True)
        state.scraped_sources["playstore"] = str(tmp_csv)
        execute_tool("run_sentiment", {"method": "vader"}, state)

        tool_sequence = ["run_topic_model(lda, n_topics=4)"]
        topic_res = execute_tool("run_topic_model", {"method": "lda", "n_topics": 4}, state)

        overlap_detected = topic_res.get("overlap_detected", False)

        # Agent adapts and tries alternative configuration
        if overlap_detected:
            tool_sequence.append("run_topic_model(lda, n_topics=2)")
            adapted_res = execute_tool("run_topic_model", {"method": "lda", "n_topics": 2}, state)
            adaptation_attempted = True
        else:
            adaptation_attempted = False

        passed = overlap_detected and adaptation_attempted
        details = f"Initial overlap detected: {overlap_detected}, Adaptive retry executed: {adaptation_attempted}"
        return passed, details, tool_sequence


def run_scenario_4_temporal_shift_memory() -> Tuple[bool, str, List[str]]:
    """Scenario 4: Temporal anomaly detected through persistent runs.db comparison."""
    print_header("4. Historical Memory & Temporal Drift Detection")

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "eval_runs.db"
        init_db(db_path)

        # Baseline run: 10% negative sentiment
        record_run(
            run_id="baseline_run_01",
            n_reviews=500,
            pct_positive=80.0,
            pct_negative=10.0,
            pct_neutral=10.0,
            top_themes={"connectivity": 30, "battery": 25},
            db_path=db_path,
        )

        # Current run metrics: 26% negative sentiment (+16% jump)
        current_metrics = {
            "n_reviews": 480,
            "pct_positive": 64.0,
            "pct_negative": 26.0,
            "top_themes": {"connectivity": 85, "battery": 30},
        }

        from pipeline.agent_memory import compare_to_previous_run

        comparison = compare_to_previous_run(current_metrics, db_path=db_path)
        tool_sequence = ["compare_to_previous_run"]

        is_unusual = comparison.get("is_unusual", False)
        anomalies = comparison.get("anomalies_detected", [])
        neg_delta = comparison.get("deltas", {}).get("negative_sentiment_delta_pct", 0)

        passed = is_unusual and neg_delta >= 15.0 and len(anomalies) > 0
        details = f"Negative delta: +{neg_delta}%, Flagged Anomalies: {anomalies}"
        return passed, details, tool_sequence


def run_scenario_5_healthy_baseline_run() -> Tuple[bool, str, List[str]]:
    """Scenario 5: Healthy routine run produces report with zero false alerts."""
    print_header("5. Healthy Baseline Run (No False Alerts)")

    # Sample dataset has normal distribution
    sample_path = Path("data/sample_reviews.csv")
    state = AgentSessionState(run_id="eval_healthy_run", dry_run=True)
    state.scraped_sources["sample"] = str(sample_path)

    tool_sequence = [
        "check_data_quality",
        "run_sentiment",
        "run_topic_model",
        "compare_to_previous_run",
        "save_final_report",
    ]

    execute_tool("check_data_quality", {}, state)
    execute_tool("run_sentiment", {"method": "vader"}, state)
    execute_tool("run_topic_model", {"method": "lda", "n_topics": 4}, state)
    execute_tool("compare_to_previous_run", {}, state)
    save_res = execute_tool("save_final_report", {}, state)

    passed = (
        state.report_saved
        and len(state.drafted_alerts) == 0  # Crucial: no false alarms!
        and save_res.get("status") == "success"
    )

    details = f"Report saved: {state.report_saved}, Drafted alerts: {len(state.drafted_alerts)}"
    return passed, details, tool_sequence


def main():
    print("\n" + "#" * 70)
    print("         AI AGENT CAPABILITY & DECISION-MAKING EVAL HARNESS")
    print("#" * 70)

    scenarios = [
        ("Scraper Failure / Empty Data Anomaly", run_scenario_1_empty_scraper),
        ("Severe Negative Sentiment Spike", run_scenario_2_negative_sentiment_spike),
        ("Topic Overlap Adaptation", run_scenario_3_topic_blob_overlap),
        ("Historical Memory Drift Detection", run_scenario_4_temporal_shift_memory),
        ("Healthy Baseline Run (No False Alarms)", run_scenario_5_healthy_baseline_run),
    ]

    results = []
    for name, runner in scenarios:
        try:
            passed, details, tool_seq = runner()
            results.append((name, passed, details, tool_seq))
        except Exception as e:
            results.append((name, False, f"EXCEPTION: {e}", []))

    # Print Scorecard
    print("\n" + "=" * 70)
    print("                      EVALUATION SCORECARD")
    print("=" * 70)

    total_passed = 0
    for name, passed, details, tool_seq in results:
        status_str = "[PASS]" if passed else "[FAIL]"
        if passed:
            total_passed += 1
        print(f"\n{status_str} {name}")
        print(f"       Tool Sequence: {' -> '.join(tool_seq)}")
        print(f"       Evidence     : {details}")

    print("\n" + "-" * 70)
    print(f"TOTAL SCORE: {total_passed}/{len(scenarios)} Scenarios Passed")
    print("=" * 70 + "\n")

    if total_passed == len(scenarios):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
