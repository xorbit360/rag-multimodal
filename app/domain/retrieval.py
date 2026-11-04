"""Recuperacion hibrida: semantica + BM25, con fusion ponderada.

BM25 implementado en el proyecto (sin servicio externo). Los puntajes se
normalizan a [0,1] por lista antes de fusionar, para que ninguna senal
domine solo por su escala. El umbral de relevancia lo aplica el caso de
uso de pregunta, no el retriever.
"""
from __future__ import annotations

import math
from collections import Counter

from app.adapters.chunking import tokenize
from app.domain.models import Chunk, RetrievedChunk
from app.domain.ports import EmbeddingPort, VectorStorePort


def bm25_scores(query: str, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
    q_tokens = tokenize(query)
    if not q_tokens or not chunks:
        return {}
    docs = [tokenize(c.text) for c in chunks]
    doc_lens = [len(d) or 1 for d in docs]
    avg_len = sum(doc_lens) / len(doc_lens)
    n = len(chunks)
    df: Counter[str] = Counter()
    for d in docs:
        for token in set(d):
            df[token] += 1
    scores: dict[str, float] = {}
    for chunk, doc_tokens, dl in zip(chunks, docs, doc_lens):
        tf = Counter(doc_tokens)
        score = 0.0
        for token in q_tokens:
            if tf[token] == 0:
                continue
            idf = math.log(1 + (n - df[token] + 0.5) / (df[token] + 0.5))
            score += idf * (tf[token] * (k1 + 1)) / (tf[token] + k1 * (1 - b + b * dl / avg_len))
        scores[chunk.id] = score
    return scores


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    max_v = max(values.values())
    if max_v <= 0:
        return {k: 0.0 for k in values}
    return {k: v / max_v for k, v in values.items()}


def hybrid_retrieve(
    question: str,
    embedding: EmbeddingPort,
    store: VectorStorePort,
    top_k: int = 5,
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4,
) -> list[RetrievedChunk]:
    q_vector = embedding.embed([question])[0]
    semantic = store.search(q_vector, top_k=max(top_k * 2, 10))
    chunks = store.all_chunks()
    keyword_raw = bm25_scores(question, chunks)
    keyword = _normalize(keyword_raw)
    merged: dict[str, RetrievedChunk] = {}
    for r in semantic:
        merged[r.chunk.id] = RetrievedChunk(
            chunk=r.chunk, semantic_score=r.semantic_score,
            keyword_score=keyword.get(r.chunk.id, 0.0))
    for chunk in chunks:
        if chunk.id in merged:
            continue
        if keyword.get(chunk.id, 0.0) > 0:
            merged[chunk.id] = RetrievedChunk(
                chunk=chunk, semantic_score=0.0, keyword_score=keyword[chunk.id])
    # OJO: el puntaje final usa el coseno CRUDO (no normalizado). Si se
    # normalizara el semantico, el mejor chunk siempre tendria 1.0 aunque
    # la pregunta fuera totalmente ajena, y el umbral de relevancia del
    # caso de uso nunca rechazaria nada. El BM25 si se normaliza, porque
    # su escala depende del tamano del corpus.
    for r in merged.values():
        r.score = semantic_weight * r.semantic_score + keyword_weight * r.keyword_score
    ranked = sorted(merged.values(), key=lambda r: r.score, reverse=True)
    return ranked[:top_k]
