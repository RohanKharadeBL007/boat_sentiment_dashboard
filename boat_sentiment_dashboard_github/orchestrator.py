"""
Master AI Agent Orchestrator for boAt Sentiment & Hardware Intelligence

Replaces the legacy static linear pipeline with an autonomous, goal-driven AI agent:
Goal stated -> Agent decides tool to call -> observes result -> decides next step ->
done when agent judges the objective is satisfied.

Key Features:
- Phase 1: Real Agent Loop using Anthropic Claude / Google Gemini / Autonomous Engine
- Phase 2: Memory Across Runs (SQLite runs.db, delta tracking, anomaly detection)
- Phase 3: Human Checkpoint Alerts (alerts drafted to pending queue, require human approval)
- Phase 4: Specialist Multi-Agent Mode (--multi-agent) with Data Quality, Analysis & Reporting agents
- Human-in-the-loop CLI tools: --list-alerts, --approve-alert <ID>, --reject-alert <ID>

Usage:
    python orchestrator.py                         # Run autonomous agent loop
    python orchestrator.py --dry-run               # Use existing data files
    python orchestrator.py --backend claude        # Use Anthropic Claude 3.5 Sonnet
    python orchestrator.py --backend gemini        # Use Google Gemini 2.5 Flash
    python orchestrator.py --multi-agent           # Run supervisor + specialist agents
    python orchestrator.py --list-alerts           # View alerts pending human approval
    python orchestrator.py --approve-alert ALT-... # Approve a drafted alert
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.agent_loop import AgentRunner
from pipeline.agent_memory import (
    init_db,
    list_pending_alerts,
    update_alert_status,
)
from pipeline.agent_tools import AgentSessionState
from pipeline.specialist_agents import MultiAgentSupervisor

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "orchestrator.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("orchestrator")


def run_agent_once(
    backend: str = "auto",
    model: str = None,
    dry_run: bool = False,
    multi_agent: bool = False,
    sources: list[str] = None,
):
    """Executes a single autonomous agent run."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info(f"============================================================")
    log.info(f"   STARTING AUTONOMOUS AGENT RUN: {run_id}")
    log.info(f"   Mode: {'Multi-Agent Supervisor' if multi_agent else 'Single Agent Goal Loop'}")
    log.info(f"   Backend: {backend} | Dry-Run: {dry_run}")
    log.info(f"============================================================")

    state = AgentSessionState(run_id=run_id, dry_run=dry_run)

    if multi_agent:
        supervisor = MultiAgentSupervisor(state=state)
        result = supervisor.execute_pipeline(target_sources=sources)
    else:
        runner = AgentRunner(backend=backend, model=model, state=state)
        result = runner.run()

    log.info(f"============================================================")
    log.info(f"   AGENT RUN COMPLETED: {run_id}")
    log.info(f"   Report Saved: {state.report_saved}")
    log.info(f"   Drafted Alerts: {len(state.drafted_alerts)}")
    if state.drafted_alerts:
        log.warning(
            f"   [HUMAN-IN-THE-LOOP] {len(state.drafted_alerts)} alert(s) require operator approval!"
        )
        log.warning(f"   Review with: python orchestrator.py --list-alerts")
    log.info(f"============================================================\n")
    return result


def handle_alerts_cli(args):
    """Handles human-in-the-loop alert review and approvals."""
    if args.list_alerts:
        alerts = list_pending_alerts()
        print("\n" + "=" * 65)
        print("          PENDING ALERTS AWAITING HUMAN APPROVAL")
        print("=" * 65)
        if not alerts:
            print("No pending alerts found. All systems healthy.")
        for a in alerts:
            print(f"\n[ALERT ID] {a['alert_id']} ({a['status']})")
            print(f"Timestamp : {a['timestamp']}")
            print(f"Severity  : {a['severity']}")
            print(f"Reason    : {a['reason']}")
            if a.get("metrics"):
                print(f"Metrics   : {json.dumps(a['metrics'], indent=2)}")
            print("-" * 65)
        print("\nTo approve: python orchestrator.py --approve-alert <ALERT_ID>")
        print("To reject : python orchestrator.py --reject-alert <ALERT_ID>\n")
        return True

    if args.approve_alert:
        ok = update_alert_status(args.approve_alert, "APPROVED")
        if ok:
            print(f"[OK] Alert {args.approve_alert} has been APPROVED for dispatch.")
        else:
            print(f"[ERROR] Alert {args.approve_alert} not found.")
        return True

    if args.reject_alert:
        ok = update_alert_status(args.reject_alert, "REJECTED")
        if ok:
            print(f"[OK] Alert {args.reject_alert} has been REJECTED/DISMISSED.")
        else:
            print(f"[ERROR] Alert {args.reject_alert} not found.")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous AI Agent Orchestrator for boAt Sentiment Intelligence."
    )
    # Operational modes
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "claude", "gemini", "cognitive_engine"],
        help="LLM Agent backend (default: auto detects Claude or Gemini key)",
    )
    parser.add_argument("--model", type=str, default=None, help="LLM model identifier")
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="Run Phase 4 specialist multi-agent supervisor system",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip web scraping and operate directly on existing datasets in data/",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["playstore", "amazon", "flipkart", "website", "twitter", "instagram"],
        help="Target data sources to evaluate",
    )
    parser.add_argument(
        "--schedule",
        choices=["once", "hourly", "daily"],
        default="once",
        help="Run once or on a recurring heartbeat schedule",
    )

    # Human-in-the-loop CLI tools
    parser.add_argument(
        "--list-alerts",
        action="store_true",
        help="Inspect all drafted alerts awaiting human approval",
    )
    parser.add_argument(
        "--approve-alert",
        type=str,
        help="Approve a drafted escalation alert for release",
    )
    parser.add_argument(
        "--reject-alert",
        type=str,
        help="Reject / dismiss a drafted escalation alert",
    )

    args = parser.parse_args()

    # If handling alert actions, process and exit
    if handle_alerts_cli(args):
        return

    # Initialize memory database
    init_db()

    # Schedule management
    interval_seconds = {"once": None, "hourly": 3600, "daily": 86400}[args.schedule]

    if interval_seconds is None:
        run_agent_once(
            backend=args.backend,
            model=args.model,
            dry_run=args.dry_run,
            multi_agent=args.multi_agent,
            sources=args.sources,
        )
    else:
        log.info(f"Running agent on a {args.schedule} schedule. Press Ctrl+C to terminate.")
        while True:
            run_agent_once(
                backend=args.backend,
                model=args.model,
                dry_run=args.dry_run,
                multi_agent=args.multi_agent,
                sources=args.sources,
            )
            log.info(f"Agent sleeping {interval_seconds}s until next observation cycle...")
            time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
