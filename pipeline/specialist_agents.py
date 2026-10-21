"""
Multi-Agent Architecture: Specialist Agents & Supervisor (Phase 4)

Implements the Orchestrator-Worker Multi-Agent Pattern:
- Data Quality Agent: Owns ingestion validation, source checks, and scraping retries.
- Analysis Agent: Owns sentiment scoring, topic modeling, and dynamic hyperparameter tuning.
- Reporting Agent: Owns historical drift comparison, root-cause drilldowns, human checkpoints, and report saving.
- Supervisor Agent: Coordinates handoffs, monitors quality gates, and aggregates intelligence.
"""

import logging
from typing import Any, Dict, Optional

from pipeline.agent_tools import (
    AgentSessionState,
    execute_tool,
)

log = logging.getLogger("specialist_agents")


class DataQualitySpecialist:
    """Specialist agent responsible for ensuring clean, valid data ingestion."""

    def __init__(self, state: AgentSessionState):
        self.state = state

    def run(self, target_sources: Optional[list[str]] = None) -> Dict[str, Any]:
        log.info("[Data Quality Agent] Auditing initial data health...")
        sources = target_sources or ["playstore"]

        # Run quality check
        audit = execute_tool("check_data_quality", {}, self.state)

        # Autonomous decision: if data missing or insufficient, attempt scraping
        if audit.get("status") == "anomaly" or audit.get("clean_reviews_ready_for_nlp", 0) < 10:
            for src in sources:
                log.info(f"[Data Quality Agent] Ingesting source: {src}")
                execute_tool("scrape_source", {"source": src}, self.state)
            audit = execute_tool("check_data_quality", {}, self.state)

        verdict = {
            "status": "APPROVED" if audit.get("clean_reviews_ready_for_nlp", 0) >= 10 else "REJECTED",
            "clean_rows": audit.get("clean_reviews_ready_for_nlp", 0),
            "source_counts": audit.get("counts_by_source", {}),
            "flags": audit.get("flags", []),
        }
        log.info(f"[Data Quality Agent] Verdict: {verdict['status']} ({verdict['clean_rows']} clean rows)")
        return verdict


class AnalysisSpecialist:
    """Specialist agent responsible for NLP sentiment classification and theme extraction."""

    def __init__(self, state: AgentSessionState):
        self.state = state

    def run(self, prefer_transformer: bool = False, n_topics: int = 6) -> Dict[str, Any]:
        log.info("[Analysis Agent] Commencing NLP sentiment scoring...")

        # Autonomous method selection based on data volume
        clean_rows = len(self.state.cleaned_df) if self.state.cleaned_df is not None else 100
        method = "transformer" if (prefer_transformer and clean_rows < 300) else "vader"
        sentiment_res = execute_tool("run_sentiment", {"method": method}, self.state)

        # Autonomous topic extraction with overlap detection
        log.info(f"[Analysis Agent] Running topic modeling with {n_topics} components...")
        topic_res = execute_tool("run_topic_model", {"method": "lda", "n_topics": n_topics}, self.state)

        # Dynamic parameter tuning if topics overlap into a giant blob
        if topic_res.get("overlap_detected"):
            log.warning("[Analysis Agent] Overlap detected in topics. Adapting to n_topics=4...")
            topic_res = execute_tool("run_topic_model", {"method": "lda", "n_topics": 4}, self.state)

        return {
            "status": "ANALYSIS_COMPLETE",
            "sentiment": sentiment_res,
            "topics": topic_res.get("topics", {}),
            "overlap_resolved": not topic_res.get("overlap_detected", False),
        }


class ReportingSpecialist:
    """Specialist agent responsible for memory tracking, human checkpoint alerts, and reporting."""

    def __init__(self, state: AgentSessionState):
        self.state = state

    def run(self) -> Dict[str, Any]:
        log.info("[Reporting Agent] Comparing against persistent historical memory...")
        memory_res = execute_tool("compare_to_previous_run", {}, self.state)
        comp = memory_res.get("comparison", {})
        anomalies = comp.get("anomalies_detected", [])

        # Check for severe complaints
        neg_pct = self.state.sentiment_stats.get("pct_negative", 0.0)
        flagged_samples = []

        if comp.get("is_unusual") or neg_pct > 25.0:
            flagged_res = execute_tool("get_flagged_reviews", {"theme": "battery", "n": 3}, self.state)
            flagged_samples = flagged_res.get("flagged_reviews", [])

            # Draft human-in-the-loop alert
            severity = "HIGH" if neg_pct > 30.0 or any("spiked" in a for a in anomalies) else "MEDIUM"
            reason = " | ".join(anomalies) if anomalies else f"Elevated negative sentiment rate: {neg_pct}%."
            log.info(f"[Reporting Agent] Drafting human checkpoint alert ({severity})...")
            execute_tool("draft_alert", {"severity": severity, "reason": reason}, self.state)

        # Save artifacts and database record
        log.info("[Reporting Agent] Finalizing report and saving dashboard artifacts...")
        save_res = execute_tool("save_final_report", {}, self.state)

        return {
            "status": "REPORTING_COMPLETE",
            "anomalies": anomalies,
            "alerts_drafted": len(self.state.drafted_alerts),
            "flagged_samples_reviewed": len(flagged_samples),
            "save_result": save_res,
        }


class MultiAgentSupervisor:
    """Orchestrates specialist agents through a coordinated supervisory workflow."""

    def __init__(self, state: Optional[AgentSessionState] = None):
        self.state = state or AgentSessionState()
        self.dq_agent = DataQualitySpecialist(self.state)
        self.analysis_agent = AnalysisSpecialist(self.state)
        self.reporting_agent = ReportingSpecialist(self.state)

    def execute_pipeline(self, target_sources: Optional[list[str]] = None) -> Dict[str, Any]:
        log.info("=== Multi-Agent Supervisor: Initializing Workflow ===")

        # Step 1: Data Quality Specialist Gate
        dq_verdict = self.dq_agent.run(target_sources=target_sources)
        if dq_verdict["status"] != "APPROVED":
            log.error("[Supervisor] Data Quality Gate failed. Halting pipeline.")
            return {
                "status": "HALTED_BY_SUPERVISOR",
                "gate": "DATA_QUALITY",
                "details": dq_verdict,
            }

        # Step 2: Analysis Specialist Execution
        analysis_verdict = self.analysis_agent.run()

        # Step 3: Reporting & Human-in-the-Loop Specialist Execution
        reporting_verdict = self.reporting_agent.run()

        log.info("=== Multi-Agent Supervisor: Workflow Complete ===")
        return {
            "status": "SUCCESS",
            "architecture": "multi_agent_supervisor",
            "data_quality": dq_verdict,
            "analysis": analysis_verdict,
            "reporting": reporting_verdict,
            "alerts_pending_human_review": len(self.state.drafted_alerts),
            "run_id": self.state.run_id,
        }
