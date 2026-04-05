# homework2_agent_system.py
# Homework 2 — Multi-agent system: 3 agents, 4 tools (2 local RAG + 2 external APIs).
#
# Run from repo root:  python homework2_agent_system.py
#
# Requires: Ollama localhost:11434, network for Frankfurter + Wikipedia.

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_H2_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_H2_DIR))

from agent_tools import (  # noqa: E402
    TOOLS_EXTERNAL_API,
    TOOLS_LOCAL_KNOWLEDGE,
    fetch_wikipedia_extract,
    get_fx_rates,
    retrieve_course_context_tool,
    search_glossary_csv,
)
from functions import agent_run, ensure_ollama_available  # noqa: E402

# Tool callables must remain in this module's globals for Ollama execution (functions.py).
MODEL = os.environ.get("OLLAMA_MODEL", "smollm2:1.7b")

DEFAULT_USER_TASK = (
    "You are preparing a short study guide on **foreign exchange**.\n"
    "1) From our **local notes**, explain what a spot rate is, how bid/ask spreads work, and "
    "why cross rates matter.\n"
    "2) Use the **glossary** to define key terms (e.g. base vs quote currency, pip).\n"
    "3) Add **live** USD→EUR, GBP, JPY rates from the API tool and relate them to the concepts "
    "in the notes.\n"
    "4) Add a **Wikipedia** intro extract on exchange rates—clearly labeled as Wikipedia, not "
    "as our bundled notes."
)


def aggregate_tool_outputs(tool_calls: Any) -> dict[str, Any]:
    """Merge every tool result from one model round (Ollama may emit multiple tool_calls)."""
    out: dict[str, Any] = {}
    if not isinstance(tool_calls, list):
        return out
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name")
        if name:
            out[name] = tc.get("output")
    return out


def run_agent_local_knowledge(user_task: str) -> tuple[str, dict[str, Any]]:
    """
    Agent 1 — Local knowledge broker.
    Tools: chunked text RAG + CSV glossary (two complementary local data sources).
    """
    role = (
        "You have two different local tools—use BOTH in this turn:\n"
        "• retrieve_course_context_tool — longer passages from bundled FX/currency .txt notes.\n"
        "• search_glossary_csv — precise rows from glossary.csv (term/definition/category).\n"
        "Derive search queries from the user’s task: concepts like spot rates, spreads, pips, "
        "cross rates, ECB, Frankfurter/API feeds, base vs quote currency. "
        "Do not pull live numeric rates here—that is Agent 2. Do not write the study guide yet; "
        "only invoke tools."
    )
    calls = agent_run(
        role=role,
        task=user_task,
        model=MODEL,
        output="tools",
        tools=TOOLS_LOCAL_KNOWLEDGE,
    )
    merged = aggregate_tool_outputs(calls)

    # Deterministic fallbacks so the pipeline stays complete if the small model skips a tool.
    if "retrieve_course_context_tool" not in merged:
        merged["retrieve_course_context_tool"] = retrieve_course_context_tool(
            query=user_task[:900], top_k=5
        )
    if "search_glossary_csv" not in merged:
        merged["search_glossary_csv"] = search_glossary_csv(
            query="spot spread pip base quote cross ECB", max_rows=8
        )

    # Strong retrieval fallback if the model’s query misses our FX corpus or scores are flat.
    def _off_topic_ml(text: str) -> bool:
        t = (text or "").lower()
        return any(
            x in t
            for x in (
                "supervised learning",
                "overfitting",
                "neural",
                "cross-validation",
                "machine learning basics",
            )
        )

    rctx = merged.get("retrieve_course_context_tool")
    if isinstance(rctx, dict):
        rq = str(rctx.get("query", ""))
        chunks = rctx.get("chunks") or []
        scores = [c.get("score", 0) for c in chunks if isinstance(c, dict)]
        weak = not scores or max(scores, default=0) == 0.0
        if weak or _off_topic_ml(rq):
            merged["retrieve_course_context_tool"] = retrieve_course_context_tool(
                query=(
                    "foreign exchange spot rate bid ask spread cross rate "
                    "Frankfurter ECB pip base quote currency"
                ),
                top_k=5,
            )

    bundle = {
        "agent": "local_knowledge",
        "tools_used": list(merged.keys()),
        "outputs": merged,
    }
    return json.dumps(bundle, indent=2), bundle


def run_agent_external_apis(user_task: str) -> tuple[str, dict[str, Any]]:
    """
    Agent 2 — External facts broker.
    Tools: Frankfurter FX + Wikipedia extract (two distinct third-party APIs).
    """
    role = (
        "You have two external tools—use BOTH:\n"
        "• get_fx_rates — ECB-based FX from Frankfurter. Use base_currency USD and "
        "quote_currencies EUR,GBP,JPY unless the user asks for others.\n"
        "• fetch_wikipedia_extract — MUST use page_title exactly: Exchange_rate "
        "(underscore; English Wikipedia article on exchange rates).\n"
        "Do not write the essay here; only call tools."
    )
    calls = agent_run(
        role=role,
        task=user_task,
        model=MODEL,
        output="tools",
        tools=TOOLS_EXTERNAL_API,
    )
    merged = aggregate_tool_outputs(calls)

    if "get_fx_rates" not in merged:
        merged["get_fx_rates"] = get_fx_rates("USD", "EUR,GBP,JPY")
    if "fetch_wikipedia_extract" not in merged:
        merged["fetch_wikipedia_extract"] = fetch_wikipedia_extract(
            page_title="Exchange_rate", max_sentences=4
        )
    else:
        wiki = merged.get("fetch_wikipedia_extract") or {}
        ex = str(wiki.get("extract", "") or "")
        bad = (
            len(ex) < 80
            or "may refer to:" in ex.lower()
            or wiki.get("page_title", "").lower() in {"live", "live!"}
        )
        if bad:
            merged["fetch_wikipedia_extract"] = fetch_wikipedia_extract(
                page_title="Exchange_rate", max_sentences=4
            )

    bundle = {
        "agent": "external_apis",
        "tools_used": list(merged.keys()),
        "outputs": merged,
    }
    return json.dumps(bundle, indent=2), bundle


def run_agent_synthesizer(user_task: str, local_json: str, external_json: str) -> str:
    """Agent 3 — No tools; integrates structured outputs with explicit attribution rules."""
    role = (
        "You are the author of a single markdown study-guide section.\n"
        "Attribution rules:\n"
        "• Facts attributed to **our local notes / glossary** must follow only JSON under "
        "outputs.retrieve_course_context_tool and outputs.search_glossary_csv.\n"
        "• **Exchange rates** must match only JSON under outputs.get_fx_rates.\n"
        "• **Wikipedia** content must be labeled explicitly as Wikipedia (from "
        "outputs.fetch_wikipedia_extract) and must not be presented as course notes.\n"
        "Structure with headings, keep the tone precise and exam-prep friendly."
    )
    task = (
        f"User task:\n{user_task}\n\n"
        f"--- JSON bundle: local knowledge agent ---\n{local_json}\n\n"
        f"--- JSON bundle: external APIs agent ---\n{external_json}\n"
    )
    return agent_run(role=role, task=task, model=MODEL, output="text", tools=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Homework 2: 3 agents, 4 tools (2× local RAG, 2× external API)"
    )
    parser.add_argument("--task", type=str, default=DEFAULT_USER_TASK, help="User task / prompt")
    args = parser.parse_args()
    user_task = (args.task or DEFAULT_USER_TASK).strip()

    ensure_ollama_available()

    sep = "=" * 72

    local_text, _ = run_agent_local_knowledge(user_task)
    print(sep)
    print("AGENT 1 — Local knowledge (tools: retrieve_course_context_tool + search_glossary_csv)")
    print(sep)
    print(local_text)
    print()

    ext_text, _ = run_agent_external_apis(user_task)
    print(sep)
    print("AGENT 2 — External APIs (tools: get_fx_rates + fetch_wikipedia_extract)")
    print(sep)
    print(ext_text)
    print()

    final = run_agent_synthesizer(user_task, local_text, ext_text)
    print(sep)
    print("AGENT 3 — Synthesizer (no tools; merges local + external JSON)")
    print(sep)
    print(final)
    print()


if __name__ == "__main__":
    main()
