# rag_retrieval.py
# RAG layer for Homework 2: retrieval from local text corpora (LAB 2 style).
# Chunked indexing + simple lexical scoring (no embeddings required for local Ollama demos).

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pandas as pd

# Corpus paths are relative to this file (repo root): data/
_H2_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_FILES = [
    _H2_DIR / "data" / "sample.txt",
    _H2_DIR / "data" / "notes_evaluation.txt",
]

GLOSSARY_CSV = _H2_DIR / "data" / "glossary.csv"


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\W+", text.lower()) if len(t) > 1]


def _chunk_text(text: str, max_chars: int = 900) -> list[str]:
    """Split on paragraphs; subdivide long blocks so retrieval stays focused."""
    raw = text.strip().split("\n\n")
    chunks: list[str] = []
    for block in raw:
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            chunks.append(block)
            continue
        for i in range(0, len(block), max_chars):
            piece = block[i : i + max_chars].strip()
            if piece:
                chunks.append(piece)
    return chunks if chunks else [text.strip()]


def _load_all_chunks(paths: list[Path]) -> list[tuple[str, str]]:
    """Return list of (source_label, chunk_text)."""
    out: list[tuple[str, str]] = []
    for p in paths:
        if not p.is_file():
            continue
        label = p.name
        with open(p, encoding="utf-8") as f:
            body = f.read()
        for ch in _chunk_text(body):
            out.append((label, ch))
    return out


_CACHE: list[tuple[str, str]] | None = None


def _chunks() -> list[tuple[str, str]]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_all_chunks(DEFAULT_CORPUS_FILES)
    return _CACHE


def _score_chunk(query: str, chunk: str) -> float:
    qt = set(_tokenize(query))
    if not qt:
        return 0.0
    ct = " ".join(_tokenize(chunk))
    score = 0.0
    for term in qt:
        if term in ct:
            # tf-like: repeat mentions increase score slightly
            score += 1.0 + math.log(1 + ct.count(term))
    return score / len(qt)


def retrieve_course_context(query: str, top_k: int = 4) -> dict:
    """
    Retrieve the most relevant text chunks from the local course corpus.

    Parameters
    ----------
    query : str
        User question or search string.
    top_k : int
        Number of chunks to return (default 4).

    Returns
    -------
    dict
        {
          "query": str,
          "top_k": int,
          "chunks": [{"source": str, "text": str, "score": float}, ...],
          "note": str,
        }
    """
    query = (query or "").strip()
    ranked: list[tuple[float, str, str]] = []
    for source, chunk in _chunks():
        s = _score_chunk(query, chunk)
        ranked.append((s, source, chunk))

    if not ranked:
        return {
            "query": query,
            "top_k": top_k,
            "chunks": [],
            "corpus_files": [str(p) for p in DEFAULT_CORPUS_FILES],
            "note": "No corpus chunks loaded; check DEFAULT_CORPUS_FILES paths.",
        }

    ranked.sort(key=lambda x: x[0], reverse=True)
    best = ranked[: max(1, top_k)]

    chunks_out = [
        {"source": src, "text": txt, "score": round(sc, 4)}
        for sc, src, txt in best
        if sc > 0
    ]
    if not chunks_out:
        # Fallback: still return highest-scoring chunks even if scores are 0
        chunks_out = [
            {"source": src, "text": txt, "score": round(sc, 4)}
            for sc, src, txt in best[:top_k]
        ]

    return {
        "query": query,
        "top_k": top_k,
        "chunks": chunks_out,
        "corpus_files": [str(p) for p in DEFAULT_CORPUS_FILES if p.is_file()],
        "note": "Lexical chunk retrieval over local text files (Homework 2 RAG).",
    }


def search_glossary(query: str, max_rows: int = 8) -> dict:
    """
    Structured retrieval over glossary.csv (term / definition / category columns).
    A row matches if the full query appears in any column, OR any query token (length > 2)
    appears in the concatenated row text (case-insensitive).
    """
    q = (query or "").strip().lower()
    max_rows = max(1, min(int(max_rows or 8), 25))
    if not GLOSSARY_CSV.is_file():
        return {
            "query": query,
            "rows": [],
            "source_file": GLOSSARY_CSV.name,
            "note": "glossary.csv not found",
        }
    df = pd.read_csv(GLOSSARY_CSV)
    if not q:
        sample = df.head(max_rows)
        return {
            "query": query,
            "rows": sample.to_dict(orient="records"),
            "source_file": GLOSSARY_CSV.name,
            "note": "Empty query; returned first rows as preview.",
        }

    tokens = [t for t in re.split(r"\W+", q) if len(t) > 2]

    def row_match(row: pd.Series) -> bool:
        hay = " ".join(str(row[c]).lower() for c in df.columns)
        if q in hay:
            return True
        return any(t in hay for t in tokens)

    matched = df[df.apply(row_match, axis=1)].head(max_rows)
    return {
        "query": query,
        "rows": matched.to_dict(orient="records"),
        "source_file": GLOSSARY_CSV.name,
        "note": "Structured glossary lookup (tabular RAG over local CSV).",
    }


def search_corpus_simple(query: str, document_path: str | None = None) -> dict:
    """
    Backward-compatible simple line search (optional helper / debugging).
    If document_path is None, uses sample.txt only.
    """
    path = Path(document_path or (_H2_DIR / "data" / "sample.txt"))
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    q = query.lower()
    matching = [line for line in lines if q in line.lower()]
    return {
        "query": query,
        "document": path.name,
        "matching_content": "".join(matching) if matching else "(no matches)",
        "num_lines": len(matching),
    }


if __name__ == "__main__":
    demo = retrieve_course_context("supervised learning algorithms", top_k=3)
    import json

    print(json.dumps(demo, indent=2))
