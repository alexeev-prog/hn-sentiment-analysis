from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from hn_sentiment_analysis.models import Story


class Embedder:
    _model = None

    @classmethod
    def get_model(cls, model_name: str):
        if cls._model is None:
            cls._model = SentenceTransformer(model_name)
        return cls._model

    @staticmethod
    def prepare_text(story: Story) -> str:
        text_parts = [story.title]

        for comment in story.comments[:3]:
            text_parts.append(comment.text)

        return " ".join(text_parts)

    @staticmethod
    def embed_batch(
        texts: List[str], model_name: str, batch_size: int = 32
    ) -> np.ndarray:
        model = Embedder.get_model(model_name)
        return model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
