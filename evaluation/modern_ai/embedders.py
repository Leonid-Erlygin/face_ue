from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class BaseTextEmbedder:
    model_name: str = "base"

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_queries(texts)


class SentenceTransformerEmbedder(BaseTextEmbedder):
    """SentenceTransformers adapter with asymmetric query/document support."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        device: Optional[str] = None,
        query_prompt_name: Optional[str] = None,
        document_prompt_name: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SentenceTransformerEmbedder requires sentence-transformers. "
                "Install requirements-modern-ai.txt."
            ) from e
        self.model_name = str(model_name)
        self.batch_size = int(batch_size)
        self.query_prompt_name = query_prompt_name
        self.document_prompt_name = document_prompt_name
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
        )

    def _encode(self, texts: Sequence[str], kind: str) -> np.ndarray:
        texts = list(map(str, texts))
        if kind == "query" and hasattr(self.model, "encode_query"):
            arr = self.model.encode_query(
                texts,
                prompt_name=self.query_prompt_name,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > self.batch_size,
            )
        elif kind == "document" and hasattr(self.model, "encode_document"):
            arr = self.model.encode_document(
                texts,
                prompt_name=self.document_prompt_name,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > self.batch_size,
            )
        else:
            arr = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > self.batch_size,
            )
        return l2_normalize(np.asarray(arr))

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "query")

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "document")


class HashingTextEmbedder(BaseTextEmbedder):
    """Dependency-light deterministic embedder for smoke tests and CI.

    It is intentionally not a scientific baseline. The production configs use
    SentenceTransformers. Hashing makes the entire experiment stack testable
    offline.
    """

    def __init__(self, n_features: int = 128, ngram_range=(1, 2)) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self.model_name = f"hashing-{n_features}"
        self.vectorizer = HashingVectorizer(
            n_features=int(n_features),
            alternate_sign=False,
            norm=None,
            ngram_range=tuple(ngram_range),
        )

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        x = self.vectorizer.transform(list(map(str, texts))).toarray()
        return l2_normalize(x)


class CachedTextEmbedder(BaseTextEmbedder):
    def __init__(self, base: BaseTextEmbedder, cache_dir: str | Path) -> None:
        self.base = base
        self.model_name = base.model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, texts: Sequence[str], kind: str) -> str:
        h = hashlib.sha256()
        h.update(self.model_name.encode())
        h.update(kind.encode())
        for text in texts:
            h.update(str(text).encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _encode(self, texts: Sequence[str], kind: str) -> np.ndarray:
        texts = tuple(map(str, texts))
        key = self._key(texts, kind)
        path = self.cache_dir / f"{key}.npz"
        if path.exists():
            return np.load(path)["embeddings"]
        if kind == "query":
            emb = self.base.encode_queries(texts)
        else:
            emb = self.base.encode_documents(texts)
        np.savez_compressed(path, embeddings=np.asarray(emb, dtype=np.float32))
        return emb

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "query")

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "document")
