"""
Autonomous AI Agent Loop (Phase 1, Phase 2, Phase 3)

Implements the goal-driven agent loop:
Goal stated -> LLM decides tool to call -> observes result -> decides again ->
done when LLM judges the goal is accomplished.

Supports:
- Anthropic Claude (Claude 3.5 Sonnet / Claude 3.7 / Haiku)
- Google Gemini (Gemini 2.5 Flash / Gemini 1.5 Pro)
- Autonomous Cognitive Engine (Deterministic Fallback / Offline / Eval simulation)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from pipeline.agent_memory import compare_to_previous_run, init_db
from pipeline.agent_tools import (
    AGENT_TOOLS_SCHEMA,
    AgentSessionState,
    execute_tool,
)

# Load environment
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

log = logging.getLogger("agent_loop")

SYSTEM_GOAL_PROMPT = """You are the Senior Product Quality & Sentiment Intelligence Autonomous Agent for boAt lifestyle.

Your goal is to autonomously orchestrate the ingestion, validation, scoring, historical comparison, and reporting of customer feedback across hardware product lines (boAt Rockerz headphones, boAt Airdopes earbuds, and boAt Wave smartwatches).

You have direct access to tools for scraping, quality auditing, sentiment scoring, topic modeling, memory comparison across previous runs, alert drafting, and final report generation.

OPERATIONAL INSTRUCTIONS:
1. Examine what data is available. Audit data quality using `check_data_quality`. If a critical source returned zero rows or failed, you can attempt to scrape it with `scrape_source`.
2. Run sentiment analysis (`run_sentiment`).
3. Run topic modeling (`run_topic_model`). If the observation notes that topics are overlapping heavily (a single giant blob), adaptively adjust parameters (e.g. adjust n_topics or switch to bertopic).
4. Check historical run memory (`compare_to_previous_run`).
5. Drill down into severe complaints using `get_flagged_reviews` if you detect significant negative volume or specific theme spikes.
6. HUMAN CHECKPOINT ESCALATION POLICY: Only call `draft_alert` if negative sentiment or an anomaly is genuinely unusual (e.g., negative delta >= 5%, review volume drop > 30%, or severe hardware failure spike). Do NOT draft alerts on healthy routine runs. All drafted alerts require human approval before dispatch.
7. Finalize and persist all artifacts using `save_final_report`.
8. Once `save_final_report` has succeeded and you have formed your conclusion, finish the conversation with a clear executive summary of your actions and findings.
"""


class AgentRunner:
    """Manages the autonomous agent loop across different LLM backends."""

    def __init__(
        self,
        backend: str = "auto",
        model: Optional[str] = None,
        max_turns: int = 15,
        state: Optional[AgentSessionState] = None,
    ):
        self.state = state or AgentSessionState()
        self.max_turns = max_turns
        self.backend = self._resolve_backend(backend)
        self.model = model or self._default_model_for_backend(self.backend)
        init_db()

    def _resolve_backend(self, requested: str) -> str:
        req = requested.lower()
        if req in ["claude", "anthropic"] and os.environ.get("ANTHROPIC_API_KEY"):
            return "claude"
        if req in ["gemini", "google"] and (os.environ.get("GOOGLE_API_KEY") or list(_ROOT.glob("client_secret*.json"))):
            return "gemini"
        if req == "auto":
            if os.environ.get("ANTHROPIC_API_KEY"):
                return "claude"
            if os.environ.get("GOOGLE_API_KEY") or list(_ROOT.glob("client_secret*.json")):
                return "gemini"
            return "cognitive_engine"
        if req == "cognitive_engine":
            return "cognitive_engine"
        log.warning(f"Requested backend '{requested}' unavailable or lacks API keys. Using autonomous cognitive engine.")
        return "cognitive_engine"

    def _default_model_for_backend(self, backend: str) -> str:
        if backend == "claude":
            return "claude-3-5-sonnet-20241022"
        elif backend == "gemini":
            return "gemini-2.5-flash"
        return "cognitive-rule-engine-v1"

    # ----------------------------------------------------------------------- #
    # Claude (Anthropic) Loop
    # ----------------------------------------------------------------------- #
    def _run_claude_loop(self) -> Dict[str, Any]:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": SYSTEM_GOAL_PROMPT}
        ]

        turns = 0
        transcript = []

        log.info(f"Starting Claude Agent Loop (Model: {self.model})")
        while turns < self.max_turns:
            turns += 1
            log.info(f"--- Turn {turns} ---")

            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                tools=AGENT_TOOLS_SCHEMA,
                messages=messages,
            )

            # Check if agent finished without tool use
            if response.stop_reason != "tool_use":
                final_text = "".join(
                    c.text for c in response.content if hasattr(c, "text")
                )
                log.info(f"Agent concluded its task: {final_text[:180]}...")
                transcript.append({"role": "assistant", "content": final_text})
                break

            # Process tool calls
            tool_calls = [c for c in response.content if c.type == "tool_use"]
            text_thoughts = "".join(
                c.text for c in response.content if hasattr(c, "text") and c.text
            )
            if text_thoughts:
                log.info(f"Agent Reasoning: {text_thoughts.strip()}")

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            # Execute tools and append observations
            tool_results = []
            for tc in tool_calls:
                log.info(f"Agent decided to invoke tool: {tc.name}")
                result = execute_tool(tc.name, tc.input, self.state)
                transcript.append({
                    "turn": turns,
                    "tool": tc.name,
                    "input": tc.input,
                    "result": result,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})

        return {
            "status": "completed",
            "backend": "claude",
            "turns_taken": turns,
            "report_saved": self.state.report_saved,
            "alerts_drafted": len(self.state.drafted_alerts),
            "transcript": transcript,
            "executive_summary": self.state.executive_summary,
        }

    # ----------------------------------------------------------------------- #
    # Gemini (Google) Loop
    # ----------------------------------------------------------------------- #
    def _run_gemini_loop(self) -> Dict[str, Any]:
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key) if api_key else genai.Client()

        # Convert schemas to Gemini format
        function_declarations = []
        for t in AGENT_TOOLS_SCHEMA:
            decl = {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }
            function_declarations.append(decl)

        chat = client.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=function_declarations)],
                system_instruction=SYSTEM_GOAL_PROMPT,
            ),
        )

        turns = 0
        transcript = []
        user_message = "Begin execution of the sentiment intelligence goal."

        while turns < self.max_turns:
            turns += 1
            log.info(f"--- Gemini Turn {turns} ---")
            response = chat.send_message(user_message)

            if not response.function_calls:
                log.info("Agent concluded task (no further tool calls).")
                transcript.append({"role": "assistant", "content": response.text})
                break

            function_responses = []
            for fc in response.function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}
                log.info(f"Gemini decided to call tool: {tool_name}")
                result = execute_tool(tool_name, tool_args, self.state)
                transcript.append({
                    "turn": turns,
                    "tool": tool_name,
                    "input": tool_args,
                    "result": result,
                })
                function_responses.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result},
                    )
                )

            user_message = function_responses

        return {
            "status": "completed",
            "backend": "gemini",
            "turns_taken": turns,
            "report_saved": self.state.report_saved,
            "alerts_drafted": len(self.state.drafted_alerts),
            "transcript": transcript,
            "executive_summary": self.state.executive_summary,
        }

    # ----------------------------------------------------------------------- #
    # Cognitive Heuristic Agent Loop (Autonomous Fallback / Eval / Offline)
    # ----------------------------------------------------------------------- #
    def _run_cognitive_engine_loop(self) -> Dict[str, Any]:
        """
        An autonomous, dynamic decision-tree agent that observes observations
        and decides the next tool call at runtime without hard-coded pipelines:
        - Evaluates check_data_quality output
        - Selects scoring algorithm
        - Evaluates topic overlap and conditionally retries
        - Compares with history in runs.db
        - Conditionally drafts alerts only on genuine anomalies
        - Saves report
        """
        transcript = []
        turns = 0
        log.info("Running Autonomous Cognitive Decision Agent Loop...")

        # Step 1: Ingest & Quality Audit
        turns += 1
        log.info("[Turn 1] Agent Action -> check_data_quality")
        quality_res = execute_tool("check_data_quality", {}, self.state)
        transcript.append({"turn": turns, "tool": "check_data_quality", "result": quality_res})

        if quality_res.get("status") == "anomaly" or quality_res.get("clean_reviews_ready_for_nlp", 0) < 10:
            turns += 1
            log.info("[Turn 2] Observation: Insufficient data. Deciding to trigger scraper for primary source.")
            scrape_res = execute_tool("scrape_source", {"source": "playstore"}, self.state)
            transcript.append({"turn": turns, "tool": "scrape_source", "result": scrape_res})
            turns += 1
            quality_res = execute_tool("check_data_quality", {}, self.state)
            transcript.append({"turn": turns, "tool": "check_data_quality", "result": quality_res})

        # Step 2: Sentiment scoring
        turns += 1
        method = "vader"
        log.info(f"[Turn {turns}] Deciding sentiment method -> {method}")
        sentiment_res = execute_tool("run_sentiment", {"method": method}, self.state)
        transcript.append({"turn": turns, "tool": "run_sentiment", "result": sentiment_res})

        # Step 3: Topic extraction & Overlap Check
        turns += 1
        log.info(f"[Turn {turns}] Extracting themes -> run_topic_model(method='lda', n_topics=6)")
        topic_res = execute_tool("run_topic_model", {"method": "lda", "n_topics": 6}, self.state)
        transcript.append({"turn": turns, "tool": "run_topic_model", "result": topic_res})

        if topic_res.get("overlap_detected"):
            turns += 1
            log.info(f"[Turn {turns}] Observation: Topic overlap detected (single blob). Retrying with n_topics=4.")
            topic_res = execute_tool("run_topic_model", {"method": "lda", "n_topics": 4}, self.state)
            transcript.append({"turn": turns, "tool": "run_topic_model", "result": topic_res})

        # Step 4: Compare against historical memory
        turns += 1
        log.info(f"[Turn {turns}] Querying persistent memory -> compare_to_previous_run")
        compare_res = execute_tool("compare_to_previous_run", {}, self.state)
        transcript.append({"turn": turns, "tool": "compare_to_previous_run", "result": compare_res})

        comp_data = compare_res.get("comparison", {})
        is_unusual = comp_data.get("is_unusual", False)
        anomalies = comp_data.get("anomalies_detected", [])

        # Check if current negative sentiment is inherently critical (> 30%)
        if sentiment_res.get("pct_negative", 0.0) >= 30.0:
            is_unusual = True
            anomalies.append(f"Absolute negative sentiment reached {sentiment_res.get('pct_negative')}%")

        # Step 5: Drill down if unusual
        if is_unusual:
            turns += 1
            log.info(f"[Turn {turns}] Anomalies detected ({len(anomalies)}). Drilling down into flagged reviews.")
            flagged_res = execute_tool("get_flagged_reviews", {"theme": "connectivity", "n": 5}, self.state)
            transcript.append({"turn": turns, "tool": "get_flagged_reviews", "result": flagged_res})

            # Step 6: Human Checkpoint Alert Draft
            turns += 1
            severity = "HIGH" if any("spiked" in a.lower() for a in anomalies) else "MEDIUM"
            reason_text = " | ".join(anomalies) if anomalies else "Unusual negative sentiment concentration."
            log.info(f"[Turn {turns}] Drafting human-in-the-loop escalation alert ({severity}).")
            alert_res = execute_tool("draft_alert", {"severity": severity, "reason": reason_text}, self.state)
            transcript.append({"turn": turns, "tool": "draft_alert", "result": alert_res})
        else:
            log.info("Observation: Metrics within normal bounds. No escalation alert required.")

        # Step 7: Finalize and Save
        turns += 1
        log.info(f"[Turn {turns}] Finalizing run -> save_final_report")
        save_res = execute_tool("save_final_report", {}, self.state)
        transcript.append({"turn": turns, "tool": "save_final_report", "result": save_res})

        return {
            "status": "completed",
            "backend": "cognitive_engine",
            "turns_taken": turns,
            "report_saved": self.state.report_saved,
            "alerts_drafted": len(self.state.drafted_alerts),
            "transcript": transcript,
            "executive_summary": self.state.executive_summary,
        }

    def run(self) -> Dict[str, Any]:
        """Main entry point to execute the agent."""
        log.info(f"Launching Agent with backend: {self.backend}")
        if self.backend == "claude":
            return self._run_claude_loop()
        elif self.backend == "gemini":
            return self._run_gemini_loop()
        else:
            return self._run_cognitive_engine_loop()
