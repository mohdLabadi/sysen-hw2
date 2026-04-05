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
    "You are preparing a study guide section.\n"
    "1) From our **local course materials**, explain supervised learning and how we judge "
    "generalization vs overfitting.\n"
    "2) Use the **glossary** to define key terms you rely on.\n"
    "3) Add **live** USD→EUR, GBP, JPY rates and a short analogy.\n"
    "4) Optionally contrast with a **Wikipedia** intro extract on supervised learning—clearly "
    "labeled as external, not as our course notes."
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
        "• retrieve_course_context_tool — longer passages from bundled .txt notes.\n"
        "• search_glossary_csv — precise rows from glossary.csv (term/definition/category).\n"
        "Derive search queries ONLY from the machine-learning / course-content parts of the task "
        "(supervised learning, generalization, overfitting, validation, glossary terms). "
        "Do not use retrieve_course_context_tool for currency rates or Wikipedia—those are handled "
        "by a later agent. Do not write the final study guide; only invoke tools."
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
            query="supervised learning overfitting validation", max_rows=8
        )

    # If the model misuses tool 1 (e.g. searches for currency text), replace with ML-focused RAG.
    def _looks_like_fx_query(text: str) -> bool:
        t = (text or "").lower()
        return any(
            x in t
            for x in ("jpy", "gbp", "eur", "usd", "exchange rate", "forex", "currency pair")
        )

    rctx = merged.get("retrieve_course_context_tool")
    if isinstance(rctx, dict):
        rq = str(rctx.get("query", ""))
        chunks = rctx.get("chunks") or []
        scores = [c.get("score", 0) for c in chunks if isinstance(c, dict)]
        weak = not scores or max(scores, default=0) == 0.0
        if _looks_like_fx_query(rq) or weak:
            merged["retrieve_course_context_tool"] = retrieve_course_context_tool(
                query=(
                    "supervised learning generalization overfitting validation "
                    "cross-validation labeled data"
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
        "• fetch_wikipedia_extract — MUST use page_title exactly: Supervised_learning "
        "(underscore, as in the English Wikipedia article title). Do not use any other title.\n"
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
            page_title="Supervised_learning", max_sentences=4
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
                page_title="Supervised_learning", max_sentences=4
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
        "• Claims about **our course** must follow only JSON under outputs.retrieve_course_context_tool "
        "and outputs.search_glossary_csv.\n"
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
