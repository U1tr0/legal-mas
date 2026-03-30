from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
from rank_bm25 import BM25Okapi

from .query_processing import tokenize_for_bm25


class BM25Index:
    """Строит BM25-индекс по текстовой колонке датафрейма."""

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "retrieval_text",
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        if text_column not in df.columns:
            raise ValueError(f"Missing required text column: {text_column}")

        self.text_column = text_column
        self.tokenizer = tokenizer or tokenize_for_bm25
        self.df = df.reset_index(drop=True).copy()
        self._texts = [self._safe_text(value) for value in self.df[text_column].tolist()]
        self._corpus_tokens = [self.tokenizer(text) for text in self._texts]
        self.bm25 = BM25Okapi(self._corpus_tokens)

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    def search(
        self,
        query: str,
        top_k: int = 10,
        query_tokens: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        if top_k <= 0:
            return self.df.iloc[0:0].copy()

        tokens = list(query_tokens) if query_tokens is not None else self.tokenizer(query)
        if not tokens:
            return self.df.iloc[0:0].copy()

        scores = self.bm25.get_scores(tokens)
        result = self.df.copy()
        result["score"] = scores
        result = result.sort_values("score", ascending=False, kind="stable")

        return result.head(min(top_k, len(result))).reset_index(drop=True)
