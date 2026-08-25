import asyncio
import time
from datetime import datetime
from functools import lru_cache
from typing import Any

import aiohttp

from hn_sentiment_analysis.hnapi.models import DateRange, StoryFilter, StorySorter
from hn_sentiment_analysis.logger import get_logger
from hn_sentiment_analysis.models import QueryParams, Story

logger = get_logger(__name__)


class HNFetcher:
    BASE_URL = "https://hn.algolia.com/api/v1/search"

    async def _fetch_page(
        self,
        session: aiohttp.ClientSession,
        page: int,
        hits_per_page: int = 100,
        query: QueryParams | None = None,
    ) -> dict[str, Any]:
        params: dict[str, str] = {
            "tags": "story",
            "hitsPerPage": str(hits_per_page),
            "page": str(page),
        }

        if query:
            params.update(query.build_dict())

        logger.debug(f"Fetching page {page} with {hits_per_page} hits per page")

        try:
            async with session.get(self.BASE_URL, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                logger.debug(f"Page {page} returned {len(data.get('hits', []))} hits")
                return data
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error fetching page {page}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching page {page}: {e}", exc_info=True)
            raise

    def _parse_story(self, raw: dict[str, Any]) -> Story:
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

        logger.debug(f"Parsed story {story.id}: '{story.title[:50]}...'")
        return story

    async def fetch_stories(
        self,
        page_limit: int = 500,
        hits_per_page: int = 100,
        query: QueryParams | None = None,
    ) -> list[Story]:
        pages = range((page_limit + hits_per_page - 1) // hits_per_page)
        timeout = aiohttp.ClientTimeout(total=30)

        logger.info(
            f"Fetching up to {page_limit} stories with {hits_per_page} per page"
        )

        start_time = time.time()

        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                self._fetch_page(session, page, hits_per_page, query) for page in pages
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            hits: list[dict[str, Any]] = []
            error_count = 0

            for idx, resp in enumerate(responses):
                if isinstance(resp, Exception):
                    logger.error(f"Error fetching page {idx}: {resp}")
                    error_count += 1
                    continue
                if isinstance(resp, dict):
                    hits.extend(resp.get("hits", []))

            logger.info(
                f"Fetched {len(hits)} hits from {len(responses)} pages, {error_count} errors"
            )

            stories = [self._parse_story(result) for result in hits]

            elapsed = time.time() - start_time
            logger.info(f"Parsed {len(stories)} stories in {elapsed:.2f}s")

            return stories

    @lru_cache(maxsize=128)
    async def get_stories(
        self,
        max_items: int = 500,
        hits_per_page: int = 100,
        sort_by: str = "score",
        sort_reverse: bool = True,
        date_range: DateRange | None = None,
        min_score: int = 0,
        author: str | None = None,
        query: QueryParams | None = None,
    ) -> list[Story]:
        logger.info(f"Getting stories with max_items={max_items}, sort_by={sort_by}")

        start_time = time.time()

        stories = await self.fetch_stories(max_items, hits_per_page, query)

        logger.info(f"Fetched {len(stories)} raw stories")

        original_count = len(stories)

        if author:
            stories = StoryFilter.by_author(stories, author)
            logger.debug(f"Filtered by author '{author}': {len(stories)} remaining")

        if min_score > 0:
            stories = StoryFilter.by_min_score(stories, min_score)
            logger.debug(f"Filtered by min_score={min_score}: {len(stories)} remaining")

        if date_range:
            stories = StoryFilter.by_date_range(stories, date_range)
            logger.debug(f"Filtered by date_range: {len(stories)} remaining")

        if sort_by == "score":
            stories = StorySorter.by_score(stories, sort_reverse)
            logger.debug(f"Sorted by score, reverse={sort_reverse}")
        elif sort_by == "date":
            stories = StorySorter.by_date(stories, sort_reverse)
            logger.debug(f"Sorted by date, reverse={sort_reverse}")

        elapsed = time.time() - start_time
        logger.info(
            f"Returning {len(stories)} stories (filtered from {original_count}) in {elapsed:.2f}s"
        )

        return stories


class HNFacade:
    def __init__(self, fetcher: HNFetcher | None = None):
        self.fetcher = fetcher or HNFetcher()
        logger.debug("HNFacade initialized")

    async def get_top_stories(
        self,
        count: int = 100,
        since: datetime | None = None,
        until: datetime | None = None,
        min_score: int = 0,
        author: str | None = None,
    ) -> list[Story]:
        logger.info(
            f"Getting top {count} stories since={since} until={until} min_score={min_score}"
        )

        start_time = time.time()

        date_range = None
        if since or until:
            date_range = DateRange(start_date=since, end_date=until)
            logger.debug(f"Date range: {date_range}")

        stories = await self.fetcher.get_stories(
            max_items=count,
            sort_by="score",
            sort_reverse=True,
            date_range=date_range,
            min_score=min_score,
            author=author,
        )

        result = stories[:count]
        elapsed = time.time() - start_time
        logger.info(f"Returned {len(result)} top stories in {elapsed:.2f}s")

        return result

    async def get_stories_by_date(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 200,
    ) -> list[Story]:
        logger.info(
            f"Getting stories by date: {start_date} to {end_date}, limit={limit}"
        )

        start_time = time.time()

        date_range = DateRange(start_date=start_date, end_date=end_date)
        stories = await self.fetcher.get_stories(
            max_items=limit,
            sort_by="date",
            sort_reverse=False,
            date_range=date_range,
        )

        elapsed = time.time() - start_time
        logger.info(f"Returned {len(stories)} stories by date in {elapsed:.2f}s")

        return stories
