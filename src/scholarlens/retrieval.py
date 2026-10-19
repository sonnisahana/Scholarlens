"""Stage 3: five interchangeable retrievers for the ablation study.

modes: tfidf | bm25 | dense | hybrid (BM25 + dense, reciprocal-rank fusion) | hybrid_rerank
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np

from . import config


def tokenize(t: str) -> list[str]:
    return re.findall(r"[\w\u0C80-\u0CFF]+", t.lower())     # \u0C80-\u0CFF = Kannada block


def rrf(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    sc: dict = defaultdict(float)
    for rl in rank_lists:
        for r, i in enumerate(rl):
            sc[i] += 1.0 / (k + r + 1)
    return sorted(sc.items(), key=lambda x: -x[1])


@lru_cache(maxsize=2)
def _st_model(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


@lru_cache(maxsize=2)
def _cross(name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name)


class Retriever:
    MODES = ("tfidf", "bm25", "dense", "hybrid", "hybrid_rerank")

    def __init__(self, chunks: list[dict], mode: str = "hybrid_rerank", pool: int = config.RETRIEVAL_POOL):
        assert mode in self.MODES, mode
        self.chunks, self.mode, self.pool = chunks, mode, pool
        self._tfidf = self._X = self._bm25 = self._emb = None
        self._masks: dict = {}
        self._e5 = "e5" in config.EMBED_MODEL.lower()

    # ---------------------------------------------------------- scorers
    def _tfidf_scores(self, q: str) -> np.ndarray:
        if self._tfidf is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                          ngram_range=(1, 2), sublinear_tf=True)
            self._X = self._tfidf.fit_transform([c["text"] for c in self.chunks])
        return (self._X @ self._tfidf.transform([q]).T).toarray().ravel()

    def _bm25_scores(self, q: str) -> np.ndarray:
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        return np.asarray(self._bm25.get_scores(tokenize(q)), dtype=float)

    def _emb_matrix(self) -> np.ndarray:
        if self._emb is None:
            key = hashlib.md5((config.EMBED_MODEL + "".join(c["chunk_id"] + c["text"][:40] for c in self.chunks))
                              .encode()).hexdigest()[:10]
            path = config.PROCESSED / f"embeddings_{key}.npy"
            if path.exists():
                self._emb = np.load(path)
            else:
                pre = "passage: " if self._e5 else ""
                self._emb = _st_model(config.EMBED_MODEL).encode(
                    [pre + c["text"] for c in self.chunks], normalize_embeddings=True,
                    batch_size=32, show_progress_bar=False)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self._emb)
        return self._emb

    def _dense_scores(self, q: str) -> np.ndarray:
        qv = _st_model(config.EMBED_MODEL).encode([("query: " if self._e5 else "") + q],
                                                  normalize_embeddings=True)[0]
        return self._emb_matrix() @ qv

    # ---------------------------------------------------------- search
    def _mask(self, scheme_id):
        if scheme_id not in self._masks:
            self._masks[scheme_id] = np.array([scheme_id is None or c["scheme_id"] == scheme_id
                                               for c in self.chunks])
        return self._masks[scheme_id]

    def _ranked(self, scores: np.ndarray, mask: np.ndarray) -> list[tuple[int, float]]:
        s = np.where(mask, scores, -np.inf)
        return [(int(i), float(s[i])) for i in np.argsort(-s) if mask[i]][: self.pool]

    def search(self, query: str, k: int = config.TOP_K, scheme_id: str | None = None,
               mode: str | None = None) -> list[dict]:
        mode = mode or self.mode
        mask = self._mask(scheme_id)
        if mode == "tfidf":
            cand = self._ranked(self._tfidf_scores(query), mask)
        elif mode == "bm25":
            cand = self._ranked(self._bm25_scores(query), mask)
        elif mode == "dense":
            cand = self._ranked(self._dense_scores(query), mask)
        else:
            lists = [[i for i, _ in self._ranked(self._bm25_scores(query), mask)],
                     [i for i, _ in self._ranked(self._dense_scores(query), mask)]]
            cand = rrf(lists)[: self.pool]
            if mode == "hybrid_rerank":
                scores = _cross(config.RERANK_MODEL).predict(
                    [(query, self.chunks[i]["text"]) for i, _ in cand])
                cand = sorted(((i, float(s)) for (i, _), s in zip(cand, scores)), key=lambda x: -x[1])
        return [dict(self.chunks[i], score=s) for i, s in cand[:k]]
