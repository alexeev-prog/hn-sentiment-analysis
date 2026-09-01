from __future__ import annotations

import asyncio
import math
import random
import time
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import aiohttp

from hn_sentiment_analysis.config import settings
from hn_sentiment_analysis.hnapi.filters import (
    CommentFilter,
    FetchParams,
    PipelineFilter,
    StoryFilter,
    algolia_constraints,
    sort_stories,
)
from hn_sentiment_analysis.logger import get_logger
from hn_sentiment_analysis.models import Comment, QueryParams, Story

logger = get_logger(__name__)

_MAX_HITS_PER_QUERY = 1000
_RETRYABLE_STATUSES = frozenset({403, 429, 500, 502, 503, 504})
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(
    total=settings.hn_request_timeout,
    sock_connect=10.0,
    sock_read=settings.hn_request_timeout,
)


class HNAPIUnavailable(Exception):
    pass


class HNDataBuilder:
    @staticmethod
    def parse_comment(raw: dict[str, Any]) -> Comment:
        comment = Comment(
            id=int(raw["objectID"]),
            text=raw.get("comment_text", ""),
            created_at=raw.get("created_at", ""),
            author=raw.get("author"),
            story_id=raw.get("story_id"),
        )
        logger.debug(f"Parsed comment {comment.id}: '{comment.text[:16]}...'")
        return comment

    @staticmethod
    def parse_story(raw: dict[str, Any]) -> Story:
        created_at = None
        if created_str := raw.get("created_at"):
            try:
                created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse date '{created_str}': {e}")

        story = Story(
            id=int(raw["objectID"]),
            title=raw.get("title", "Untitled"),
            url=raw.get("url"),
            author=raw.get("author", "unknown"),
            score=raw.get("points", 0),
            created_at=created_at,
            tags=raw.get("_tags", []),
        )
        logger.debug(f"Parsed story {story.id}: '{story.title[:16]}...'")
        return story


class HNFetcher:
    SEARCH_URL = "https://hn.algolia.com/api/v1/search"
    SEARCH_BY_DATE_URL = "https://hn.algolia.com/api/v1/search_by_date"

    def __init__(self, search_by_date: bool = False):
        self.search_by_date = search_by_date
        self._semaphore: asyncio.Semaphore | None = None
        self._consecutive_failures = 0
        self._unavailable = False

    @property
    def _url(self) -> str:
        return self.SEARCH_BY_DATE_URL if self.search_by_date else self.SEARCH_URL

    def _reset_health(self) -> None:
        self._consecutive_failures = 0
        self._unavailable = False
        self._semaphore = asyncio.Semaphore(settings.hn_request_concurrency)

    def _check_available(self) -> None:
        if self._unavailable:
            raise HNAPIUnavailable("HN Algolia API unavailable (circuit open)")

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures < settings.hn_failure_threshold:
            return
        if not self._unavailable:
            self._unavailable = True
            logger.error(
                f"HN Algolia API marked unavailable after "
                f"{self._consecutive_failures} consecutive failed requests "
                "(rate limit or outage?): aborting remaining requests"
            )

    async def _attempt_request(
        self, session: aiohttp.ClientSession, params: dict[str, Any]
    ) -> dict[str, Any]:
        semaphore = self._semaphore
        assert semaphore is not None
        async with semaphore:
            self._check_available()
            if settings.hn_request_delay > 0:
                await asyncio.sleep(settings.hn_request_delay)
            async with session.get(
                self._url, params=params, timeout=_REQUEST_TIMEOUT
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def _get_json(
        self, session: aiohttp.ClientSession, params: dict[str, Any]
    ) -> dict[str, Any]:
        self._check_available()

        attempts = settings.hn_request_retries + 1
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                data = await self._attempt_request(session, params)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                self._record_failure()
                last_exc = exc
                status = getattr(exc, "status", None)
                if status is not None and status not in _RETRYABLE_STATUSES:
                    raise
                if self._unavailable:
                    raise HNAPIUnavailable(
                        "HN Algolia API unavailable (circuit open)"
                    ) from exc
                if attempt < attempts:
                    delay = settings.hn_retry_backoff * attempt + random.uniform(0, 0.5)
                    logger.warning(
                        f"HN API error, status={status} "
                        f"(attempt {attempt}/{attempts}): "
                        f"retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                continue
            self._consecutive_failures = 0
            return data

        assert last_exc is not None
        raise last_exc

    async def _fetch_page(
        self,
        session: aiohttp.ClientSession,
        page: int,
        hits_per_page: int = 100,
        tag_type: str = "story",
        query: QueryParams | None = None,
    ) -> dict[str, Any]:
        if query is None:
            query = QueryParams(
                tags=[tag_type],
                hitsPerPage=hits_per_page,
                page=page,
            )
        return await self._get_json(session, query.build_dict())

    async def _fetch_raw_pages(
        self,
        session: aiohttp.ClientSession,
        pages: Sequence[int],
        hits_per_page: int,
        tag_type: str = "story",
        query: QueryParams | None = None,
    ) -> list[Any]:
        tasks = [
            self._fetch_page(
                session,
                page,
                hits_per_page,
                tag_type,
                query.model_copy(update={"page": page}) if query else None,
            )
            for page in pages
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        errors = 0
        for page, resp in zip(pages, responses):
            if isinstance(resp, HNAPIUnavailable):
                continue
            if isinstance(resp, BaseException):
                logger.error(f"Failed to fetch page {page}: {resp}")
                errors += 1
        if errors:
            logger.warning(f"{errors}/{len(pages)} pages failed")
        return list(responses)

    @staticmethod
    def _extract_hits(responses: Sequence[Any]) -> list[dict[str, Any]]:
        return [
            hit
            for response in responses
            if isinstance(response, dict)
            for hit in response.get("hits", [])
        ]

    async def _fetch_pages(
        self,
        session: aiohttp.ClientSession,
        max_items: int,
        hits_per_page: int,
        tag_type: str = "story",
        query: QueryParams | None = None,
    ) -> list[dict[str, Any]]:
        pages = range((max_items + hits_per_page - 1) // hits_per_page)
        responses = await self._fetch_raw_pages(
            session, pages, hits_per_page, tag_type, query
        )
        for resp in responses:
            if isinstance(resp, HNAPIUnavailable):
                raise resp
        return self._extract_hits(responses)

    @staticmethod
    def _search_query(
        query: QueryParams | None,
        numeric: Sequence[str],
        tags: Sequence[str],
        hits_per_page: int,
    ) -> QueryParams:
        base = query or QueryParams(tags=["story"])
        base_numeric = [base.numeric_filters] if base.numeric_filters else []
        merged_numeric = base_numeric + list(numeric)
        merged_tags = list(dict.fromkeys((base.tags or ["story"]) + list(tags)))
        return QueryParams(
            query=base.query,
            tags=merged_tags,
            numeric_filters=",".join(merged_numeric) if merged_numeric else None,
            filters=base.filters,
            hitsPerPage=hits_per_page,
            page=0,
        )

    async def _collect_by_date(
        self,
        session: aiohttp.ClientSession,
        target: int,
        story_filters: Sequence[StoryFilter],
        query: QueryParams | None,
        hits_per_page: int,
        max_pages: int,
    ) -> list[Story]:
        numeric, tag_extras = algolia_constraints(story_filters)
        composite = PipelineFilter(list(story_filters))
        pages_cap = max(1, _MAX_HITS_PER_QUERY // hits_per_page)
        collected: list[Story] = []
        cursor: int | None = None
        pages_done = 0

        while len(collected) < target and pages_done < max_pages:
            pages_needed = math.ceil((target - len(collected)) / hits_per_page)
            batch = min(pages_needed, pages_cap, max_pages - pages_done)
            batch_numeric = numeric + (
                [f"created_at_i<{cursor}"] if cursor is not None else []
            )
            search = self._search_query(query, batch_numeric, tag_extras, hits_per_page)

            responses = await self._fetch_raw_pages(
                session, range(batch), hits_per_page, "story", search
            )
            pages_done += batch
            raw_hits = self._extract_hits(responses)
            if not raw_hits:
                break

            parsed = [HNDataBuilder.parse_story(hit) for hit in raw_hits]
            collected.extend(composite.apply(parsed))

            timestamps = [
                hit["created_at_i"]
                for hit in raw_hits
                if isinstance(hit.get("created_at_i"), int)
            ]
            if not timestamps:
                logger.warning("Hits without created_at_i: collection stopped")
                break
            new_cursor = min(timestamps)
            if cursor is not None and new_cursor >= cursor:
                break
            cursor = new_cursor

        if len(collected) < target:
            if self._unavailable:
                reason = "HN API marked unavailable"
            elif pages_done >= max_pages:
                reason = (
                    f"page budget max_pages={max_pages} reached "
                    "(raise max_pages / HN_MAX_PAGES)"
                )
            else:
                reason = "filters or API limit reached"
            logger.warning(
                f"Collected only {len(collected)}/{target} stories "
                f"in {pages_done} pages: {reason}"
            )
        return collected[:target]

    async def _collect_by_relevance(
        self,
        session: aiohttp.ClientSession,
        target: int,
        story_filters: Sequence[StoryFilter],
        query: QueryParams | None,
        hits_per_page: int,
        max_pages: int,
    ) -> list[Story]:
        numeric, tag_extras = algolia_constraints(story_filters)
        composite = PipelineFilter(list(story_filters))
        search = self._search_query(query, numeric, tag_extras, hits_per_page)
        pages_cap = max(1, _MAX_HITS_PER_QUERY // hits_per_page)
        pages = min(math.ceil(target / hits_per_page), pages_cap, max_pages)

        responses = await self._fetch_raw_pages(
            session, range(pages), hits_per_page, "story", search
        )
        raw_hits = self._extract_hits(responses)
        parsed = [HNDataBuilder.parse_story(hit) for hit in raw_hits]
        stories = composite.apply(parsed)
        if len(stories) < target:
            logger.warning(
                f"Collected only {len(stories)}/{target} stories: Algolia serves "
                f"at most {_MAX_HITS_PER_QUERY} hits per query "
                f"(use search_by_date=True for deeper collection)"
            )
        return stories[:target]

    async def _collect_stories(
        self,
        session: aiohttp.ClientSession,
        target: int,
        story_filters: Sequence[StoryFilter],
        query: QueryParams | None,
        hits_per_page: int,
        max_pages: int,
    ) -> list[Story]:
        collector = (
            self._collect_by_date if self.search_by_date else self._collect_by_relevance
        )
        return await collector(
            session, target, story_filters, query, hits_per_page, max_pages
        )

    async def _fetch_comments_for_story(
        self,
        session: aiohttp.ClientSession,
        story_id: int,
        max_comments: int,
        comment_filter: CommentFilter,
    ) -> list[Comment]:
        query = QueryParams(
            tags=["comment"],
            hitsPerPage=min(max_comments, 100),
            page=0,
            filters=f"story_id={story_id}",
        )

        raw_comments = await self._fetch_pages(
            session,
            max_comments,
            min(max_comments, 100),
            "comment",
            query,
        )

        comments = comment_filter.apply_to_comments(
            [HNDataBuilder.parse_comment(raw) for raw in raw_comments]
        )
        logger.debug(f"Fetched {len(comments)} comments for story {story_id}")
        return comments

    async def fetch_stories(
        self,
        max_items: int = 500,
        story_filters: Sequence[StoryFilter] | None = None,
        comment_filter: CommentFilter | None = None,
        query: QueryParams | None = None,
        hits_per_page: int = 100,
        fetch_comments: bool = True,
        max_comments_per_story: int = 10,
        max_pages: int = 50,
    ) -> list[Story]:
        start_time = time.time()
        self._reset_health()
        filters = list(story_filters or [])
        comment_filter = comment_filter or CommentFilter()

        async with aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT) as session:
            stories = await self._collect_stories(
                session, max_items, filters, query, hits_per_page, max_pages
            )

            if self._unavailable:
                logger.error(
                    "HN Algolia API unavailable: fetch aborted early, "
                    "comment fetching is skipped for this run"
                )
            elif fetch_comments and stories:
                comment_tasks = [
                    self._fetch_comments_for_story(
                        session,
                        story.id,
                        max_comments_per_story,
                        comment_filter,
                    )
                    for story in stories
                ]
                comments_batch = await asyncio.gather(
                    *comment_tasks, return_exceptions=True
                )
                self._attach_comments(stories, comments_batch)

            elapsed = time.time() - start_time
            logger.info(f"Fetched {len(stories)} stories in {elapsed:.2f}s")
            return stories

    @staticmethod
    def _attach_comments(stories: list[Story], batch: Sequence[Any]) -> None:
        failures = [item for item in batch if isinstance(item, BaseException)]
        if failures:
            logger.warning(
                f"Comments unavailable for {len(failures)}/{len(stories)} stories "
                f"({failures[0]})"
            )
        for story, comments in zip(stories, batch):
            if isinstance(comments, list):
                story.comments = comments


class HNFacade:
    def __init__(
        self, fetcher: HNFetcher | None = None, search_by_date: bool = False
    ) -> None:
        self.fetcher = fetcher or HNFetcher(search_by_date=search_by_date)

    async def get_stories(self, fetch: FetchParams | None = None) -> list[Story]:
        fetch = fetch or FetchParams()
        start_time = time.time()

        comment_filter = fetch.comment_filter or CommentFilter(
            min_length=settings.comment_min_length,
            max_length=settings.comment_max_length,
        )

        mode = "date walk" if self.fetcher.search_by_date else "relevance"
        logger.info(
            f"Fetching stories from HN ({len(fetch.story_filters)} filters; "
            f"sort={fetch.sort_by}; {mode})"
        )

        stories = await self.fetcher.fetch_stories(
            max_items=fetch.count,
            story_filters=fetch.story_filters,
            comment_filter=comment_filter,
            query=fetch.query,
            hits_per_page=fetch.hits_per_page,
            fetch_comments=fetch.fetch_comments,
            max_comments_per_story=fetch.max_comments_per_story,
            max_pages=fetch.max_pages,
        )

        stories = sort_stories(stories, fetch.sort_by, fetch.sort_reverse)
        stories = stories[: fetch.count]

        elapsed = time.time() - start_time
        logger.info(f"Returned {len(stories)} stories in {elapsed:.2f}s")

        return stories
