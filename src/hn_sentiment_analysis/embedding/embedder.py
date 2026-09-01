from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from hn_sentiment_analysis.config import settings
from hn_sentiment_analysis.logger import get_logger

logger = get_logger(__name__)


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name or settings.embedding_model)
        self._batch_size = batch_size or settings.embedding_batch_size

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        logger.debug(f"Embedding {len(texts)} documents (batch={self._batch_size})")
        return self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
