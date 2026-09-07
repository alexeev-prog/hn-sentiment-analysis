# clustering/clusterer.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import umap  # type: ignore
from sklearn.cluster import HDBSCAN  # type: ignore

from hn_sentiment_analysis.config import settings
from hn_sentiment_analysis.logger import get_logger
from hn_sentiment_analysis.models import Story, StoryCluster
from hn_sentiment_analysis.utils import extract_top_terms, title_tokens

logger = get_logger(__name__)

NOISE_LABEL = -1
_RANDOM_STATE = 42
_KNN = 5
_ASSIGN_CHUNK = 2000


@dataclass(slots=True)
class ClusterResult:
    labels: np.ndarray
    points: np.ndarray | None = None


class BaseClusterer(ABC):
    @abstractmethod
    def fit_predict(self, embeddings: np.ndarray) -> ClusterResult: ...


class UMAPHDBSCANClusterer(BaseClusterer):
    def __init__(
        self,
        n_components: int | None = None,
        n_neighbors: int | None = None,
        min_cluster_size: int | None = None,
        selection_method: str | None = None,
        min_dist: float | None = None,
        min_samples: int | None = None,
    ) -> None:
        self._n_components = n_components or settings.umap_n_components
        self._n_neighbors = n_neighbors or settings.umap_n_neighbors
        self._min_cluster_size = min_cluster_size or settings.cluster_min_size
        self._selection_method = selection_method or settings.cluster_selection_method
        self._min_dist = settings.umap_min_dist if min_dist is None else min_dist
        self._min_samples = (
            settings.cluster_min_samples if min_samples is None else min_samples
        )
        self._assign_outliers = settings.cluster_assign_outliers
        self._outlier_threshold = settings.cluster_outlier_threshold

    def fit_predict(self, embeddings: np.ndarray) -> ClusterResult:
        n_samples = len(embeddings)
        min_cluster_size = self._effective_cluster_size(n_samples)
        if n_samples < min_cluster_size:
            logger.warning(
                f"{n_samples} samples < min_cluster_size={min_cluster_size}: "
                "clustering skipped"
            )
            return ClusterResult(labels=np.full(n_samples, NOISE_LABEL, dtype=np.int32))

        n_neighbors = self._effective_neighbors(n_samples)

        reducer = umap.UMAP(
            n_components=min(self._n_components, n_samples - 1),
            n_neighbors=n_neighbors,
            min_dist=self._min_dist,
            metric="cosine",
            random_state=_RANDOM_STATE,
        )
        reduced = reducer.fit_transform(embeddings.astype(np.float32))

        labels = (
            HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=self._min_samples or None,
                cluster_selection_method=self._selection_method,
                copy=False,
            )
            .fit_predict(reduced)
            .astype(np.int32)
        )

        noise_before = int((labels == NOISE_LABEL).sum())
        if self._assign_outliers:
            labels = self._assign_outliers_to_clusters(embeddings, labels)
        points = reduced[:, :2] if reduced.shape[1] >= 2 else None

        n_clusters = len(set(labels.tolist()) - {NOISE_LABEL})
        logger.info(
            f"Clustering done: {n_clusters} clusters, "
            f"{int((labels == NOISE_LABEL).sum())}/{n_samples} noise points "
            f"(was {noise_before} before outlier assignment; "
            f"min_cluster_size={min_cluster_size}, n_neighbors={n_neighbors}, "
            f"min_dist={self._min_dist}, method={self._selection_method}, "
            f"min_samples={self._min_samples or 'auto'})"
        )
        if n_clusters == 0:
            logger.warning(
                "No clusters found: try lower cluster_min_size or more stories"
            )
        return ClusterResult(labels=labels, points=points)

    def _effective_cluster_size(self, n_samples: int) -> int:
        if settings.cluster_auto_min_size:
            return max(self._min_cluster_size, max(5, n_samples // 200))
        return self._min_cluster_size

    def _effective_neighbors(self, n_samples: int) -> int:
        return max(2, min(self._n_neighbors, n_samples - 1, max(3, n_samples // 4)))

    def _assign_outliers_to_clusters(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        noise_positions = np.flatnonzero(labels == NOISE_LABEL)
        clustered_positions = np.flatnonzero(labels != NOISE_LABEL)
        if noise_positions.size == 0 or clustered_positions.size == 0:
            return labels

        data = embeddings.astype(np.float32)
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        data = data / norms
        reference = data[clustered_positions]

        k = min(_KNN, clustered_positions.size)
        required_votes = max(1, (k + 1) // 2)
        labels = labels.copy()
        assigned = 0

        for start in range(0, noise_positions.size, _ASSIGN_CHUNK):
            chunk = noise_positions[start : start + _ASSIGN_CHUNK]
            similarities = data[chunk] @ reference.T
            top = np.argpartition(-similarities, k - 1, axis=1)[:, :k]
            top_similarities = np.take_along_axis(similarities, top, axis=1)

            for row in range(len(chunk)):
                votes: dict[int, list[float]] = {}
                for col in range(k):
                    neighbor_label = int(labels[clustered_positions[top[row, col]]])
                    votes.setdefault(neighbor_label, []).append(
                        float(top_similarities[row, col])
                    )
                best_label, best_votes = max(
                    votes.items(), key=lambda item: len(item[1])
                )
                if (
                    len(best_votes) >= required_votes
                    and sum(best_votes) / len(best_votes) >= self._outlier_threshold
                ):
                    labels[chunk[row]] = best_label
                    assigned += 1

        logger.info(
            f"Outlier assignment: {assigned}/{noise_positions.size} noise points "
            f"joined clusters via {k}-NN majority "
            f"(cosine >= {self._outlier_threshold})"
        )
        return labels


def group_into_clusters(
    stories: Sequence[Story], labels: Sequence[int]
) -> tuple[list[StoryCluster], list[Story]]:
    clusters: dict[int, list[Story]] = {}
    outliers: list[Story] = []

    for story, label in zip(stories, labels):
        story.cluster_label = int(label)
        if label == NOISE_LABEL:
            outliers.append(story)
        else:
            clusters.setdefault(int(label), []).append(story)

    corpus_freq: Counter[str] = Counter()
    for story in stories:
        corpus_freq.update(title_tokens(story.title))

    grouped = [
        StoryCluster(
            label=label,
            stories=members,
            top_terms=extract_top_terms(
                [story.title for story in members], corpus_freq
            ),
        )
        for label, members in clusters.items()
    ]
    grouped.sort(key=lambda cluster: cluster.total_score, reverse=True)
    return grouped, outliers
