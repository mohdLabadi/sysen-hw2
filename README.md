# Homework 2 — AI Agent System with RAG and Tools

Self-contained folder for the cumulative assignment: **multi-agent orchestration**, **RAG over multiple local sources**, and **function calling** to **several** tools (local + external APIs). Course lab folders are **not** required.

Use this file for the **Documentation** section of your Canvas `.docx`. The **reflective writing** (3+ paragraphs, your own words, not AI-generated) is separate in Word.

---

## What makes this submission “complete”

| Requirement | How it is met |
|-------------|----------------|
| **2–3 agents** | Three agents: (1) local knowledge broker, (2) external-API broker, (3) synthesizer. |
| **RAG** | **Two** local retrieval modes: chunked `.txt` notes + **tabular** `glossary.csv` (structured RAG). Multiple corpus files under `data/`. |
| **Function calling / tools** | **Four** tools: two local data sources + **two** external REST APIs (Frankfurter FX, Wikipedia MediaWiki). |
| **Nuanced orchestration** | Each tool-using agent may receive **multiple** `tool_calls` per model round; the code **aggregates all outputs** (not only the first). Deterministic **fallbacks** run if a small model omits a tool so demos stay reproducible. |

---

## File map

| Path | Purpose |
|------|---------|
| `homework2_agent_system.py` | **Main orchestration** — three agents, wiring, JSON bundles. |
| `rag_retrieval.py` | **RAG implementation** — chunking, lexical scoring, `search_glossary()`. |
| `agent_tools.py` | **Tool implementations + Ollama JSON schemas** for all four tools. |
| `functions.py` | Ollama HTTP client (`agent` / `agent_run`). |
| `data/sample.txt`, `notes_evaluation.txt` | Text corpora for chunk retrieval. |
| `data/glossary.csv` | Tabular glossary for `search_glossary_csv`. |
| `01_ollama.py` | Optional background `ollama serve`. |
| `requirements.txt` | Dependencies. |

---

## System architecture (agent roles)

1. **Agent 1 — Local knowledge broker**  
   - **Tools:** `retrieve_course_context_tool`, `search_glossary_csv`.  
   - **Intent:** Pull prose context from notes + definitional rows from the glossary (different data shapes, same “grounded in local files” story).

2. **Agent 2 — External API broker**  
   - **Tools:** `get_fx_rates`, `fetch_wikipedia_extract`.  
   - **Intent:** Live numeric FX data + optional encyclopedia intro; both are third-party and must be labeled in the final text.

3. **Agent 3 — Synthesizer**  
   - **Tools:** none.  
   - **Intent:** One markdown study-guide section with strict attribution: local JSON vs FX JSON vs Wikipedia.

---

## RAG data sources and search behavior

| Source | File(s) | Behavior |
|--------|---------|----------|
| Course notes (text) | `data/sample.txt`, `data/notes_evaluation.txt` | Paragraph-aware chunks, lexical overlap scoring, top‑k chunks with scores. |
| Glossary (tabular) | `data/glossary.csv` | Rows match if any significant query token hits concatenated columns (or full query substring). |

Extend retrieval by adding more `.txt` paths in `DEFAULT_CORPUS_FILES` (`rag_retrieval.py`) or more rows to `glossary.csv`.

---

## Tool functions (for your docx table)

| Tool name | Purpose | Parameters | Returns (summary) |
|-----------|---------|------------|-------------------|
| `retrieve_course_context_tool` | Chunked retrieval from local `.txt` corpora. | `query`, optional `top_k` | Dict with `chunks` (source, text, score), `corpus_files`, etc. |
| `search_glossary_csv` | Structured lookup in `glossary.csv`. | `query`, optional `max_rows` | Dict with `rows` (list of records), `source_file`. |
| `get_fx_rates` | Live FX rates. | `base_currency`, `quote_currencies` | Dict with `rates`, `rates_table`, `date`, `api`. |
| `fetch_wikipedia_extract` | English Wikipedia intro extract. | `page_title`, optional `max_sentences` | Dict with `extract`, `article_url`, attribution `note`. |

Schemas: `agent_tools.py`.

---

## Technical details

| Topic | Detail |
|-------|--------|
| LLM | Ollama `http://localhost:11434`, default model `smollm2:1.7b` — override with `OLLAMA_MODEL`. |
| APIs | `https://api.frankfurter.app/latest` (no key); `https://en.wikipedia.org/w/api.php` (no key; **User-Agent** set per Wikimedia guidance). |
| Python | `requests`, `pandas`, `tabulate` — see `requirements.txt`. |

---

## Usage (recommended: virtual environment)

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull smollm2:1.7b
python homework2_agent_system.py
```

Custom task:

```bash
python homework2_agent_system.py --task "Your instructions here"
```

**Screenshots:** capture Agent 1 JSON (both local tools), Agent 2 JSON (both external tools), Agent 3 markdown — plus optional `python rag_retrieval.py` for RAG-only output.

---

## Git links (example paths)

- Orchestration / main: `homework2_agent_system.py`
- RAG: `rag_retrieval.py`
- Tools: `agent_tools.py`
