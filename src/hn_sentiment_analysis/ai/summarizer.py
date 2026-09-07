from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from openai import APIConnectionError, AsyncOpenAI, RateLimitError

from hn_sentiment_analysis.config import settings
from hn_sentiment_analysis.logger import get_logger
from hn_sentiment_analysis.models import ClusterSummary, StoryCluster
from hn_sentiment_analysis.utils import strip_html, truncate

logger = get_logger(__name__)

_SENTIMENTS = ("positive", "negative", "mixed", "neutral")
_JSON_SPEC = (
    '[{"index": <cluster number>, "title": "<short heading, 3-7 words>", '
    '"description": "<2-3 sentences>", '
    '"sentiment": "<positive|negative|mixed|neutral>"}]'
)


class BaseSummarizer(ABC):
    @abstractmethod
    async def summarize(self, cluster: StoryCluster) -> ClusterSummary: ...

    async def summarize_many(
        self, clusters: Sequence[StoryCluster]
    ) -> list[ClusterSummary]:
        return list(await asyncio.gather(*(self.summarize(c) for c in clusters)))


class LLMClusterSummarizer(BaseSummarizer):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_concurrent: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._model = model or settings.ai_default_model
        self._max_retries = (
            settings.ai_max_retries if max_retries is None else max_retries
        )
        self._client: AsyncOpenAI | None = None
        self._semaphore: asyncio.Semaphore | None = None

        key = api_key if api_key is not None else settings.ai_api_key
        if key:
            self._client = AsyncOpenAI(
                api_key=key,
                base_url=base_url or settings.ai_base_url,
                timeout=settings.ai_request_timeout,
            )
            self._semaphore = asyncio.Semaphore(
                settings.ai_max_concurrent if max_concurrent is None else max_concurrent
            )
        else:
            logger.warning(
                "ai_api_key is not set: cluster summaries will use fallback titles"
            )

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    async def summarize(self, cluster: StoryCluster) -> ClusterSummary:
        return (await self._summarize_batch([cluster]))[0]

    async def summarize_many(
        self, clusters: Sequence[StoryCluster]
    ) -> list[ClusterSummary]:
        clusters = list(clusters)
        if not clusters:
            return []
        size = max(1, settings.ai_batch_size)
        batches = [clusters[i : i + size] for i in range(0, len(clusters), size)]
        results = await asyncio.gather(*(self._summarize_batch(b) for b in batches))
        return [summary for batch in results for summary in batch]

    async def _summarize_batch(self, batch: list[StoryCluster]) -> list[ClusterSummary]:
        if self._client is None or self._semaphore is None:
            return [self._fallback(cluster) for cluster in batch]

        labels = [cluster.label for cluster in batch]
        async with self._semaphore:
            try:
                return await self._request_batch_summary(batch)
            except Exception as ex:
                logger.error(f"LLM summary failed for clusters {labels}: {ex}")
                return [self._fallback(cluster) for cluster in batch]

    async def _request_batch_summary(
        self, batch: list[StoryCluster]
    ) -> list[ClusterSummary]:
        assert self._client is not None
        labels = [cluster.label for cluster in batch]
        messages = self._build_messages(batch)
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 2):
            try:
                logger.info(
                    f"Requesting LLM summary for clusters {labels} (attempt {attempt})"
                )
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore
                    temperature=settings.ai_temperature,
                    max_tokens=settings.ai_max_tokens * len(batch),
                )
                content = response.choices[0].message.content or ""
                return self._parse_batch(content, batch)
            except (APIConnectionError, RateLimitError) as exc:
                last_exc = exc
                delay = settings.ai_retry_backoff * attempt
                logger.warning(
                    f"Clusters {labels}: attempt {attempt} failed ({last_exc}); "
                    f"retrying in {delay:.0f}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.info(
                    f"LLM summary succeeded for clusters {labels} on attempt {attempt}"
                )

        raise RuntimeError(
            f"All {self._max_retries + 1} attempts failed for clusters {labels}"
        ) from last_exc

    def _build_messages(self, batch: list[StoryCluster]) -> list[dict[str, str]]:
        sections = []
        for position, cluster in enumerate(batch, start=1):
            lines = [
                f"Cluster #{position}: {cluster.size} related posts, "
                f"{cluster.total_comments} comments, "
                f"{cluster.total_score} points total"
            ]
            top_stories = sorted(
                cluster.stories, key=lambda s: s.score or 0, reverse=True
            )
            for story in top_stories[: settings.ai_max_stories_in_prompt]:
                lines.append(f"- ({story.score} points) {story.title}")
                for comment in story.comments[: settings.ai_max_comments_in_prompt]:
                    snippet = truncate(
                        strip_html(comment.text), settings.ai_prompt_comment_chars
                    )
                    if snippet:
                        lines.append(f"    * {snippet}")
            sections.append("\n".join(lines))

        system = (
            "You are an expert analyst of Hacker News discussions. You receive "
            f"{len(batch)} cluster(s) of related posts with comment snippets. "
            "Answer ONLY with a valid minified JSON array containing exactly one "
            f"object per cluster, in input order, in the format {_JSON_SPEC}. "
            f"Write all fields in {settings.ai_summary_language}. For each "
            "cluster: 'index' repeats its cluster number; 'title' must name the "
            "concrete shared topic (a product, technology, company or event) — "
            "never a vague heading like 'Tech Discussions'; 'description' must "
            "state what unites the posts, the community's dominant opinion, and "
            "any notable disagreement or concern; 'sentiment' is the overall "
            "tone of the discussion: 'positive', 'negative', 'mixed' when "
            "opinions clearly split, or 'neutral' for factual discussions. "
            "No markdown, no extra keys, no text outside the JSON array."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(sections)},
        ]

    def _parse_batch(
        self, content: str, batch: list[StoryCluster]
    ) -> list[ClusterSummary]:
        data = self._extract_json(content)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array: {content[:200]!r}")

        items = [item for item in data if isinstance(item, dict)]
        by_position: dict[int, dict[str, Any]] = {}
        for item in items:
            position = item.get("index")
            if isinstance(position, int) and 1 <= position <= len(batch):
                by_position[position] = item
        if not by_position and len(items) == len(batch):
            by_position = dict(zip(range(1, len(batch) + 1), items))

        summaries = []
        for position, cluster in enumerate(batch, start=1):
            item = by_position.get(position)  # type: ignore
            if item is None:
                summaries.append(self._fallback(cluster))
                continue
            try:
                summaries.append(self._summary_from(item))
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    f"Malformed LLM summary for cluster {cluster.label}: fallback"
                )
                summaries.append(self._fallback(cluster))
        return summaries

    def _summary_from(self, item: dict[str, Any]) -> ClusterSummary:
        title = str(item["title"]).strip()
        description = str(item["description"]).strip()
        if not title or not description:
            raise ValueError("empty title or description")
        sentiment = str(item.get("sentiment", "")).strip().lower()
        return ClusterSummary(
            title=title,
            description=description,
            sentiment=sentiment if sentiment in _SENTIMENTS else None,
            model=self._model,
        )

    @staticmethod
    def _extract_json(content: str) -> Any:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"[\[{].*[\]}]", content, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _fallback(cluster: StoryCluster) -> ClusterSummary:
        top_title = cluster.stories[0].title if cluster.stories else "Empty cluster"
        return ClusterSummary(
            title=truncate(f"Cluster #{cluster.label}: {top_title}", 80),
            description=(
                f"A group of {cluster.size} related posts (LLM summary unavailable)."
            ),
        )
