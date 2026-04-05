# Homework 2 — AI Agent System with RAG and Tools

This repository is the **technical + documentation** deliverable. Copy sections below into your Canvas **`.docx`** as needed. Your **written reflection** (3+ paragraphs, your own words) is separate in Word.

---

## Quick start (run the system)

1. **Install [Ollama](https://ollama.com)** and start it (`ollama serve` in a terminal, or run `python 01_ollama.py` once).
2. Open a terminal in the **repository root** (the folder that contains `homework2_agent_system.py`).
3. Create and activate a virtual environment, then install Python packages:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Pull the default model: `ollama pull smollm2:1.7b` (or set another model with the `OLLAMA_MODEL` environment variable).
5. Run: `python homework2_agent_system.py`

**No API keys** are required. You **do** need internet for live FX rates and Wikipedia. If you see `ModuleNotFoundError: pandas`, activate the venv and run `pip install -r requirements.txt` again.

Optional: `python homework2_agent_system.py --task "Your custom instructions"`  
Optional RAG-only demo: `python rag_retrieval.py`

---

## 1. System architecture

**Workflow (what runs, in order):**

1. **Agent 1 — Local knowledge broker**  
   Uses **function calling** with two tools: `retrieve_course_context_tool` (reads your bundled `.txt` notes) and `search_glossary_csv` (reads `glossary.csv`). The model is prompted to call both; the script also merges all tool outputs and applies simple fallbacks if a small model skips a tool.

2. **Agent 2 — External API broker**  
   Uses **function calling** with two tools: `get_fx_rates` (live rates from Frankfurter) and `fetch_wikipedia_extract` (English Wikipedia intro). Defaults: USD vs EUR, GBP, JPY; Wikipedia article `Exchange_rate` unless the model supplies a valid call.

3. **Agent 3 — Synthesizer**  
   **No tools.** Takes the JSON from Agents 1 and 2 and writes **one markdown** answer, with rules so local notes/glossary, live rates, and Wikipedia are not mixed up.

**Main entry point:** `homework2_agent_system.py`  
**RAG logic:** `rag_retrieval.py`  
**Tool definitions + HTTP helpers:** `agent_tools.py`  
**Ollama HTTP wrapper:** `functions.py`

---

## 2. RAG data source and search functions

**Text corpus (chunked retrieval)**  
- **Files:** `data/sample.txt`, `data/notes_evaluation.txt` (FX/currency topics: spot rates, spreads, cross rates, APIs, etc.).  
- **How it works:** Implemented in `rag_retrieval.py` as `retrieve_course_context(query, top_k)`. Text is split into paragraph-style **chunks**, each chunk is **scored** by lexical overlap with the query, and the **top‑k** chunks are returned with `{source, text, score}`.  
- **Config:** List of files is `DEFAULT_CORPUS_FILES` in `rag_retrieval.py` (add more `.txt` paths there to extend the knowledge base).

**Tabular glossary (structured retrieval)**  
- **File:** `data/glossary.csv` (columns: `term`, `definition`, `category`).  
- **How it works:** `search_glossary(query, max_rows)` in `rag_retrieval.py` loads the CSV with pandas and returns rows where the query matches as a **substring** of any column **or** where **significant words** from the query appear in the row text (token-style match).

**Tool wrappers for the LLM:** `retrieve_course_context_tool` and `search_glossary_csv` in `agent_tools.py` call the functions above.

---

## 3. Tool functions

| Name | Purpose | Parameters | What it returns |
|------|---------|------------|-----------------|
| `retrieve_course_context_tool` | Retrieve top text chunks from local `.txt` corpora. | `query` (string), `top_k` (int, optional, default 4) | Dict with `chunks` (list of `{source, text, score}`), `corpus_files`, `query`, `note`. |
| `search_glossary_csv` | Look up rows in `glossary.csv`. | `query` (string), `max_rows` (int, optional) | Dict with `rows` (list of dicts), `source_file`, `note`. |
| `get_fx_rates` | Latest FX rates from Frankfurter (ECB-based). | `base_currency` (e.g. `USD`), `quote_currencies` (comma-separated, e.g. `EUR,GBP,JPY`) | Dict with `rates`, `rates_table`, `date`, `base`, `api`. |
| `fetch_wikipedia_extract` | First sentences of a Wikipedia article (intro). | `page_title` (e.g. `Exchange_rate`), `max_sentences` (optional) | Dict with `extract`, `article_url`, `page_title`, `note` (says it is third-party). |

Ollama **JSON schemas** for these tools are in `agent_tools.py`.

---

## 4. Technical details

| Topic | Detail |
|-------|--------|
| **API keys** | **None.** Frankfurter and Wikipedia are public HTTP APIs. |
| **Endpoints** | Ollama chat: `http://localhost:11434` · Frankfurter: `https://api.frankfurter.app/latest` · Wikipedia: `https://en.wikipedia.org/w/api.php` |
| **LLM** | Default model `smollm2:1.7b`. Override: `export OLLAMA_MODEL=your-model` (macOS/Linux) before running. |
| **Python packages** | `requests`, `pandas`, `tabulate` — pinned in `requirements.txt`. |
| **Repo layout** | See table below. |

| File / folder | Role |
|---------------|------|
| `homework2_agent_system.py` | Main script: three agents, prompts, fallbacks. |
| `rag_retrieval.py` | Chunking, scoring, `search_glossary`. |
| `agent_tools.py` | Tool implementations + tool JSON for Ollama. |
| `functions.py` | `agent` / `agent_run` → Ollama HTTP API. |
| `data/*.txt`, `data/glossary.csv` | Local RAG sources (edit these to change what retrieval sees). |
| `requirements.txt` | `pip install -r requirements.txt` |
| `01_ollama.py` | Optional: starts `ollama serve` in the background. |

---

## 5. Usage instructions (install, data, run)

**Install dependencies**

```bash
cd /path/to/sysen-hw2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Data sources** — Already included under `data/`. To customize: edit the `.txt` files and `glossary.csv`, or add `.txt` paths in `DEFAULT_CORPUS_FILES` inside `rag_retrieval.py`. No database or embedding server is required.

**API keys** — Not used; nothing to configure.

**Ollama** — Install Ollama, then either keep `ollama serve` running or use `python 01_ollama.py`. Pull a model: `ollama pull smollm2:1.7b`.

**Run**

```bash
source .venv/bin/activate
python homework2_agent_system.py
```

**Screenshots for the report:** Use the printed blocks **AGENT 1**, **AGENT 2**, and **AGENT 3**. Optionally run `python rag_retrieval.py` for a RAG-only screenshot.

---

## Git links (for Canvas)

Replace with your GitHub **blob** URLs on `main`:

- **Orchestration / main system:** `homework2_agent_system.py`
- **RAG implementation:** `rag_retrieval.py`
- **Tools / function definitions:** `agent_tools.py`

Example pattern: `https://github.com/<user>/<repo>/blob/main/homework2_agent_system.py`
