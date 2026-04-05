# agent_tools.py
# Tool definitions for Homework 2: local RAG (text + CSV), external REST APIs.

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from rag_retrieval import retrieve_course_context, search_glossary

# --- Tool 1: Chunked text RAG (local files under data/) ---


def retrieve_course_context_tool(query: str, top_k: int = 4) -> dict:
    k = int(top_k) if top_k else 4
    return retrieve_course_context(query, top_k=max(1, min(k, 12)))


TOOL_RETRIEVE_COURSE_CONTEXT: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_course_context_tool",
        "description": (
            "Retrieve top-ranked text chunks from bundled course notes (.txt files) using "
            "lexical scoring. Use for conceptual questions aligned with your local corpus."
        ),
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query or paraphrased user question.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to return (1–12). Default 4.",
                },
            },
        },
    },
}


# --- Tool 2: Tabular / structured RAG (local CSV glossary) ---


def search_glossary_csv(query: str, max_rows: int = 8) -> dict:
    """LLM-facing name; delegates to rag_retrieval.search_glossary."""
    mr = int(max_rows) if max_rows else 8
    return search_glossary(query, max_rows=max(1, min(mr, 25)))


TOOL_SEARCH_GLOSSARY_CSV: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_glossary_csv",
        "description": (
            "Search the local glossary.csv for term/definition/category rows matching the query. "
            "Use for crisp definitions and vocabulary tied to supervised learning and evaluation."
        ),
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Term or keywords to look up in the glossary.",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Max rows to return (1–25). Default 8.",
                },
            },
        },
    },
}


# --- Tool 3: External API — FX rates (Frankfurter) ---


def get_fx_rates(base_currency: str = "USD", quote_currencies: str = "EUR,GBP,JPY") -> dict:
    url = "https://api.frankfurter.app/latest"
    params = {
        "from": base_currency.strip().upper(),
        "to": quote_currencies.replace(" ", ""),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    rates = data.get("rates") or {}
    df = pd.DataFrame(
        {
            "currency": list(rates.keys()),
            "units_of_quote_per_1_base": list(rates.values()),
        }
    )
    return {
        "api": "Frankfurter",
        "base": data.get("base", base_currency),
        "date": data.get("date"),
        "rates_table": df.to_string(index=False),
        "rates": rates,
    }


TOOL_GET_FX_RATES: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_fx_rates",
        "description": (
            "Fetch current foreign exchange rates (ECB-based) from api.frankfurter.app. "
            "Use for any real-world currency numbers in the answer."
        ),
        "parameters": {
            "type": "object",
            "required": ["base_currency", "quote_currencies"],
            "properties": {
                "base_currency": {"type": "string", "description": "ISO 4217 base, e.g. USD"},
                "quote_currencies": {
                    "type": "string",
                    "description": "Comma-separated ISO codes, e.g. EUR,GBP,JPY",
                },
            },
        },
    },
}


# --- Tool 4: External API — Wikipedia extracts (MediaWiki) ---


def fetch_wikipedia_extract(
    page_title: str = "Supervised_learning",
    max_sentences: int = 4,
) -> dict:
    """
    Short plain-text extract from English Wikipedia (intro sentences only).
    Attribution: not course material; third-party encyclopedia.
    """
    if not (page_title or "").strip():
        page_title = "Supervised_learning"
    ms = int(max_sentences) if max_sentences else 4
    ms = max(1, min(ms, 10))
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": page_title.strip(),
        "exsentences": ms,
    }
    headers = {"User-Agent": "SysEnHomework2/1.0 (educational; local agent demo)"}
    r = requests.get(url, params=params, timeout=30, headers=headers)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    if page.get("missing"):
        return {
            "page_title": page_title,
            "api": "Wikipedia",
            "error": "page_not_found",
            "extract": "",
        }
    title = page.get("title", page_title)
    extract = page.get("extract", "") or ""
    slug = title.replace(" ", "_")
    return {
        "page_title": title,
        "api": "Wikipedia",
        "extract": extract.strip(),
        "article_url": f"https://en.wikipedia.org/wiki/{slug}",
        "note": "Third-party summary—not part of the bundled course notes.",
    }


TOOL_FETCH_WIKIPEDIA_EXTRACT: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_wikipedia_extract",
        "description": (
            "Fetch a short Wikipedia intro extract for a given article title (English). "
            "Use for optional external context; clearly separate from local course materials."
        ),
        "parameters": {
            "type": "object",
            "required": ["page_title"],
            "properties": {
                "page_title": {
                    "type": "string",
                    "description": 'Article title, e.g. "Supervised_learning" or "Machine learning"',
                },
                "max_sentences": {
                    "type": "integer",
                    "description": "Intro sentences (1–10). Default 4.",
                },
            },
        },
    },
}


TOOLS_LOCAL_KNOWLEDGE = [
    TOOL_RETRIEVE_COURSE_CONTEXT,
    TOOL_SEARCH_GLOSSARY_CSV,
]

TOOLS_EXTERNAL_API = [
    TOOL_GET_FX_RATES,
    TOOL_FETCH_WIKIPEDIA_EXTRACT,
]

ALL_TOOLS = TOOLS_LOCAL_KNOWLEDGE + TOOLS_EXTERNAL_API
