from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from hn_sentiment_analysis.ai.summarizer import BaseSummarizer, LLMClusterSummarizer
from hn_sentiment_analysis.clustering.clusterer import (
    BaseClusterer,
    ClusterResult,
    UMAPHDBSCANClusterer,
    group_into_clusters,
)
from hn_sentiment_analysis.embedding import BaseEmbedder, SentenceTransformerEmbedder
from hn_sentiment_analysis.hnapi.fetcher import HNFacade
from hn_sentiment_analysis.hnapi.filters import FetchParams
from hn_sentiment_analysis.logger import get_logger
from hn_sentiment_analysis.models import PipelineResult, Story, StoryCluster
from hn_sentiment_analysis.reporting.html_report import HTMLReportBuilder
from hn_sentiment_analysis.serializer import JSONReportBuilder

logger = get_logger(__name__)


@dataclass(slots=True)
class PipelineParams:
    fetch: FetchParams = field(default_factory=FetchParams)
    build_report: bool = True
    build_json: bool = True
    json_path: Path | None = None
    json_include_embedding: bool = False


class StorySource(Protocol):
    async def get_stories(self, fetch: FetchParams) -> list[Story]: ...


class ReportBuilder(Protocol):
    def build(self, result: PipelineResult) -> Path: ...


class HNAnalysisPipeline:
    def __init__(
        self,
        fetcher: StorySource,
        embedder: BaseEmbedder,
        clusterer: BaseClusterer,
        summarizer: BaseSummarizer,
        report_builder: ReportBuilder,
        json_builder: JSONReportBuilder | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._embedder = embedder
        self._clusterer = clusterer
        self._summarizer = summarizer
        self._report_builder = report_builder
        self._json_builder = json_builder

    async def run(self, params: PipelineParams | None = None) -> PipelineResult:
        params = params or PipelineParams()
        started = time.perf_counter()

        stories = await self._extract(params.fetch)
        if len(stories) < 2:
            logger.warning(f"Only {len(stories)} stories fetched — clustering skipped")
            result = PipelineResult(
                outliers=stories,
                total_stories=len(stories),
                elapsed_seconds=time.perf_counter() - started,
            )
            if params.build_report:
                self._report_builder.build(result)
            if params.build_json:
                self._build_json(result, params)
            return result

        self._embed(stories)
        clusters, outliers = self._cluster(stories)
        await self._summarize(clusters)

        result = PipelineResult(
            clusters=clusters,
            outliers=outliers,
            total_stories=len(stories),
            elapsed_seconds=time.perf_counter() - started,
        )
        if params.build_report:
            self._report_builder.build(result)
        if params.build_json:
            self._build_json(result, params)

        logger.info(
            f"Pipeline finished in {result.elapsed_seconds:.1f}s: "
            f"{len(clusters)} clusters, {len(outliers)} outliers"
        )
        return result

    def _build_json(self, result: PipelineResult, params: PipelineParams) -> None:
        builder = self._json_builder or JSONReportBuilder(
            output_path=params.json_path,
            include_embedding=params.json_include_embedding,
        )
        builder.build(result)

    async def _extract(self, fetch: FetchParams) -> list[Story]:
        stage = time.perf_counter()
        stories = await self._fetcher.get_stories(fetch)
        logger.info(
            f"[extract] {len(stories)} stories in {time.perf_counter() - stage:.2f}s"
        )
        return stories

    def _embed(self, stories: list[Story]) -> None:
        stage = time.perf_counter()
        embeddings = self._embedder.embed_documents(
            [story.embedding_text for story in stories]
        )
        for story, vector in zip(stories, embeddings):
            story.embedding = vector.tolist()
        logger.info(
            f"[embed] {len(stories)} vectors, dim={embeddings.shape[1]} "
            f"in {time.perf_counter() - stage:.2f}s"
        )

    def _cluster(self, stories: list[Story]) -> tuple[list[StoryCluster], list[Story]]:
        stage = time.perf_counter()
        matrix = np.asarray([story.embedding for story in stories], dtype=np.float32)
        outcome = self._clusterer.fit_predict(matrix)
        clusters, outliers = group_into_clusters(stories, outcome.labels.tolist())
        self._assign_positions(stories, outcome)
        logger.info(
            f"[cluster] {len(clusters)} clusters in {time.perf_counter() - stage:.2f}s"
        )
        return clusters, outliers

    @staticmethod
    def _assign_positions(stories: list[Story], outcome: ClusterResult) -> None:
        if outcome.points is None:
            return
        for story, (x, y) in zip(stories, outcome.points.tolist()):
            story.pos_x = float(x)
            story.pos_y = float(y)

    async def _summarize(self, clusters: list[StoryCluster]) -> None:
        if not clusters:
            return
        stage = time.perf_counter()
        summaries = await self._summarizer.summarize_many(clusters)
        for cluster, summary in zip(clusters, summaries):
            cluster.summary = summary
        logger.info(
            f"[summarize] {len(clusters)} summaries "
            f"in {time.perf_counter() - stage:.2f}s"
        )


def build_default_pipeline(
    search_by_date: bool = False,
    report_path: str | Path | None = None,
    json_path: str | Path | None = None,
    json_include_embedding: bool = False,
) -> HNAnalysisPipeline:
    logger.info("Start default pipeline")

    return HNAnalysisPipeline(
        fetcher=HNFacade(search_by_date=search_by_date),
        embedder=SentenceTransformerEmbedder(),
        clusterer=UMAPHDBSCANClusterer(),
        summarizer=LLMClusterSummarizer(),
        report_builder=HTMLReportBuilder(report_path),
        json_builder=JSONReportBuilder(
            output_path=json_path,
            include_embedding=json_include_embedding,
        ),
    )
