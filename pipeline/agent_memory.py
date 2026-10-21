"""
Agent Memory & Human-in-the-Loop Storage System (Phase 2 & Phase 3)

Maintains persistent run history in SQLite (`runs.db`) across executions,
enabling temporal awareness:
- Tracking delta in volume and sentiment week-over-week
- Flagging anomalous spikes in negative themes (e.g., connectivity or battery)
- Managing Human-in-the-Loop pending alerts requiring approval before release
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path("data/runs.db")
OUTPUTS_DIR = Path("outputs")


def init_db(db_path: Path = DB_PATH) -> None:
    """Initializes runs and pending_alerts tables if not already present."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                n_reviews INTEGER NOT NULL,
                pct_positive REAL NOT NULL,
                pct_negative REAL NOT NULL,
                pct_neutral REAL NOT NULL,
                top_themes_json TEXT,
                source_breakdown_json TEXT,
                alerts_count INTEGER DEFAULT 0,
                metadata_json TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_alerts (
                alert_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                severity TEXT NOT NULL,
                reason TEXT NOT NULL,
                metrics_json TEXT,
                status TEXT NOT NULL DEFAULT 'PENDING_APPROVAL',
                approved_at TEXT,
                approved_by TEXT,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
            """
        )
        conn.commit()


def record_run(
    run_id: str,
    n_reviews: int,
    pct_positive: float,
    pct_negative: float,
    pct_neutral: float,
    top_themes: Optional[Dict[str, Any]] = None,
    source_breakdown: Optional[Dict[str, Any]] = None,
    alerts_count: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Path = DB_PATH,
) -> None:
    """Inserts a completed run record into persistent storage."""
    init_db(db_path)
    now_str = datetime.now().isoformat()
    top_themes_json = json.dumps(top_themes or {})
    source_json = json.dumps(source_breakdown or {})
    meta_json = json.dumps(metadata or {})

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, timestamp, n_reviews, pct_positive, pct_negative,
                pct_neutral, top_themes_json, source_breakdown_json, alerts_count, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                now_str,
                n_reviews,
                round(pct_positive, 2),
                round(pct_negative, 2),
                round(pct_neutral, 2),
                top_themes_json,
                source_json,
                alerts_count,
                meta_json,
            ),
        )
        conn.commit()


def get_previous_run(db_path: Path = DB_PATH, exclude_run_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves the most recent prior run."""
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if exclude_run_id:
            cursor.execute(
                "SELECT * FROM runs WHERE run_id != ? ORDER BY timestamp DESC LIMIT 1",
                (exclude_run_id,),
            )
        else:
            cursor.execute("SELECT * FROM runs ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)


def get_run_history(limit: int = 10, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves the recent run history."""
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(r) for r in cursor.fetchall()]


def compare_to_previous_run(
    current_metrics: Dict[str, Any], db_path: Path = DB_PATH
) -> Dict[str, Any]:
    """
    Compares current run metrics against the last recorded run.
    Identifies significant volume drops, sentiment shifts, and theme-level anomalies.
    """
    prev = get_previous_run(db_path)
    if not prev:
        return {
            "has_baseline": False,
            "message": "No previous run history found in database. Establishing baseline with current run.",
            "deltas": {},
            "anomalies_detected": [],
        }

    prev_n = prev["n_reviews"]
    curr_n = current_metrics.get("n_reviews", 0)
    vol_change_pct = round(((curr_n - prev_n) / prev_n) * 100, 1) if prev_n > 0 else 0.0

    prev_pos = prev["pct_positive"]
    curr_pos = current_metrics.get("pct_positive", 0.0)
    pos_delta = round(curr_pos - prev_pos, 2)

    prev_neg = prev["pct_negative"]
    curr_neg = current_metrics.get("pct_negative", 0.0)
    neg_delta = round(curr_neg - prev_neg, 2)

    anomalies = []
    # Check 1: Sharp volume drop (> 30% drop)
    if vol_change_pct < -30:
        anomalies.append(f"Review ingestion dropped by {abs(vol_change_pct)}% compared to baseline.")

    # Check 2: Spike in negative sentiment (> +5% increase)
    if neg_delta >= 5.0:
        anomalies.append(
            f"Negative sentiment spiked by +{neg_delta}% (from {prev_neg}% to {curr_neg}%)."
        )

    # Check 3: Theme-level shifts if themes JSON available
    theme_shifts = {}
    try:
        prev_themes = json.loads(prev["top_themes_json"] or "{}")
        curr_themes = current_metrics.get("top_themes", {})
        for theme, count in curr_themes.items():
            if theme in prev_themes:
                prev_cnt = prev_themes[theme]
                shift = count - prev_cnt
                if shift > 10:
                    theme_shifts[theme] = f"+{shift} reviews vs previous run"
    except Exception:
        pass

    return {
        "has_baseline": True,
        "previous_run_id": prev["run_id"],
        "previous_timestamp": prev["timestamp"],
        "deltas": {
            "volume_change_pct": vol_change_pct,
            "positive_sentiment_delta_pct": pos_delta,
            "negative_sentiment_delta_pct": neg_delta,
        },
        "theme_shifts": theme_shifts,
        "anomalies_detected": anomalies,
        "is_unusual": len(anomalies) > 0,
    }


# --------------------------------------------------------------------------- #
# Human-in-the-Loop Pending Alerts (Phase 3)
# --------------------------------------------------------------------------- #

def save_alert_draft(
    alert_id: str,
    run_id: str,
    severity: str,
    reason: str,
    metrics: Optional[Dict[str, Any]] = None,
    db_path: Path = DB_PATH,
) -> Dict[str, Any]:
    """
    Saves an alert in PENDING_APPROVAL status.
    Writes to both SQLite and outputs/pending_alerts.json for human review.
    Does NOT auto-send to external channels.
    """
    init_db(db_path)
    now_str = datetime.now().isoformat()
    metrics_json = json.dumps(metrics or {})

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO pending_alerts (
                alert_id, run_id, timestamp, severity, reason, metrics_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, 'PENDING_APPROVAL')
            """,
            (alert_id, run_id, now_str, severity.upper(), reason, metrics_json),
        )
        conn.commit()

    # Sync to JSON file for easy human inspection and CLI review
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    alerts = list_pending_alerts(db_path=db_path)
    alerts_file = OUTPUTS_DIR / "pending_alerts.json"
    alerts_file.write_text(json.dumps(alerts, indent=2))

    summary_file = OUTPUTS_DIR / "pending_alerts.txt"
    lines = ["=== PENDING ALERTS (AWAITING HUMAN APPROVAL) ==="]
    for a in alerts:
        lines.append(
            f"[{a['status']}] Alert ID: {a['alert_id']} | Severity: {a['severity']} | Run: {a['run_id']}"
        )
        lines.append(f"Reason: {a['reason']}")
        lines.append(f"Timestamp: {a['timestamp']}")
        lines.append("-" * 60)
    summary_file.write_text("\n".join(lines))

    return {
        "alert_id": alert_id,
        "status": "PENDING_APPROVAL",
        "action_required": "Human approval required before dispatching external notification.",
        "inspect_file": str(alerts_file),
    }


def list_pending_alerts(
    status: str = "PENDING_APPROVAL", db_path: Path = DB_PATH
) -> List[Dict[str, Any]]:
    """Returns all alerts with the specified status."""
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM pending_alerts WHERE status = ? ORDER BY timestamp DESC",
            (status,),
        )
        rows = cursor.fetchall()
        alerts = []
        for r in rows:
            d = dict(r)
            if d.get("metrics_json"):
                try:
                    d["metrics"] = json.loads(d["metrics_json"])
                except Exception:
                    pass
            alerts.append(d)
        return alerts


def update_alert_status(
    alert_id: str,
    new_status: str,
    approved_by: str = "human_operator",
    db_path: Path = DB_PATH,
) -> bool:
    """Updates the status of an alert (e.g. APPROVED, REJECTED)."""
    init_db(db_path)
    now_str = datetime.now().isoformat()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE pending_alerts
            SET status = ?, approved_at = ?, approved_by = ?
            WHERE alert_id = ?
            """,
            (new_status.upper(), now_str, approved_by, alert_id),
        )
        conn.commit()
        updated = cursor.rowcount > 0

    if updated:
        # Re-sync JSON file
        alerts = list_pending_alerts(status="PENDING_APPROVAL", db_path=db_path)
        (OUTPUTS_DIR / "pending_alerts.json").write_text(json.dumps(alerts, indent=2))

    return updated
