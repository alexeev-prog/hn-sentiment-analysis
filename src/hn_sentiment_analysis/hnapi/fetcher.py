import asyncio
import time
from datetime import datetime
from typing import Any

import aiohttp

from hn_sentiment_analysis.config import settings
from hn_sentiment_analysis.hnapi.filters import (
    AuthorFilter,
    CommentFilter,
    DateRange,
    DateRangeFilter,
    KeywordFilter,
    MinScoreFilter,
    PipelineFilter,
    SortBy,
    StoryFilter,
    StorySorter,
)
from hn_sentiment_analysis.logger import get_logger
from hn_sentiment_analysis.models import Comment, QueryParams, Story

logger = get_logger(__name__)


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
    __BASE_URL = "https://hn.algolia.com/api/v1/search"
    __SEARCH_BY_DATE_URL = "https://hn.algolia.com/api/v1/search_by_date"

    def __init__(
        self,
        search_by_date: bool = False,
        story_filter: PipelineFilter | None = None,
        comment_filter: CommentFilter | None = None,
    ):
        self.story_filter = story_filter or PipelineFilter([])
        self.comment_filter = comment_filter or CommentFilter()
        self.search_by_date = search_by_date

    @property
    def BASE_URL(self):
        if self.search_by_date:
            return self.__SEARCH_BY_DATE_URL
        else:
            return self.__BASE_URL

    async def _fetch_comments_by_story_id(
        self,
        session: aiohttp.ClientSession,
        page: int,
        hits_per_page: int = 100,
        query: QueryParams | None = None,
    ) -> dict[str, Any]:
        if query is None:
            query = QueryParams(
                tags=["comment"],
                hitsPerPage=hits_per_page,
                page=page,
            )

        params: dict[str, Any] = query.build_dict()

        logger.debug(
            f"Fetching comments page {page} with {hits_per_page} hits per page"
        )

        try:
            async with session.get(self.BASE_URL, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                logger.debug(
                    f"Comments page {page} returned {len(data.get('hits', []))} hits"
                )

                return data
        except aiohttp.ClientError as e:
            logger.error(
                f"HTTP error fetching comments page {page}: {e}", exc_info=True
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error fetching comments page {page}: {e}", exc_info=True
            )
            raise

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

        params = query.build_dict()

        try:
            async with session.get(self.BASE_URL, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return data
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error fetching page {page}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching page {page}: {e}")
            raise

    async def _fetch_pages(
        self,
        session: aiohttp.ClientSession,
        max_items: int,
        hits_per_page: int,
        tag_type: str = "story",
        query: QueryParams | None = None,
    ) -> list[dict[str, Any]]:
        # ИСПРАВЛЕНИЕ 2: Убрали + 1, чтобы не было запроса лишней страницы
        pages = range(0, (max_items + hits_per_page - 1) // hits_per_page)

        # ИСПРАВЛЕНИЕ 1: Подставляем текущую страницу в копию query
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

        all_hits = []
        errors = 0

        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                logger.error(f"Error fetching page {i}: {resp}")
                errors += 1
            elif isinstance(resp, dict):
                all_hits.extend(resp.get("hits", []))

        if errors:
            logger.warning(f"Failed to fetch {errors} pages")

        return all_hits

    async def _fetch_comments_for_story(
        self,
        session: aiohttp.ClientSession,
        story_id: int,
        max_comments: int = 100,
    ) -> list[Comment]:
        query = QueryParams(
            tags=["comment"],
            hitsPerPage=min(max_comments, 100),
            # ИСПРАВЛЕНИЕ 3: У Algolia страница начинается с 0, а не с 1
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

        comments = [HNDataBuilder.parse_comment(raw) for raw in raw_comments]
        comments = self.comment_filter.apply_to_comments(comments)

        logger.debug(f"Fetched {len(comments)} comments for story {story_id}")
        return comments

    async def fetch_stories(
        self,
        max_items: int = 500,
        hits_per_page: int = 100,
        query: QueryParams | None = None,
        fetch_comments: bool = True,
        max_comments_per_story: int = 10,
    ) -> list[Story]:
        start_time = time.time()

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            raw_stories = await self._fetch_pages(
                session, max_items, hits_per_page, "story", query
            )

            stories = [HNDataBuilder.parse_story(raw) for raw in raw_stories]
            stories = self.story_filter.apply(stories)

            if fetch_comments and stories:
                comment_tasks = [
                    self._fetch_comments_for_story(
                        session, story.id, max_comments_per_story
                    )
                    for story in stories
                ]

                comments_batch = await asyncio.gather(
                    *comment_tasks, return_exceptions=True
                )

                for story, comments in zip(stories, comments_batch):
                    if isinstance(comments, list):
                        story.comments = comments
                    else:
                        logger.warning(
                            f"Failed to fetch comments for story {story.id}: {comments}"
                        )

            elapsed = time.time() - start_time
            logger.info(f"Fetched {len(stories)} stories in {elapsed:.2f}s")

            return stories


class HNFacade:
    def __init__(self, fetcher: HNFetcher | None = None, search_by_date: bool = False):
        self.fetcher = fetcher or HNFetcher(search_by_date=search_by_date)

    async def get_stories(
        self,
        count: int = 100,
        since: datetime | None = None,
        until: datetime | None = None,
        min_score: int = 0,
        author: str | None = None,
        keyword: str | None = None,
        sort_by: SortBy | str = SortBy.SCORE,
        sort_reverse: bool = True,
    ) -> list[Story]:
        start_time = time.time()

        story_filters: list[StoryFilter] = []

        if since or until:
            date_range = DateRange(start_date=since, end_date=until)
            story_filters.append(DateRangeFilter(date_range))

        if min_score > 0:
            story_filters.append(MinScoreFilter(min_score))

        if author:
            story_filters.append(AuthorFilter(author))

        if keyword:
            story_filters.append(KeywordFilter(keyword))

        logger.info(
            f"Start parsing stories from HN ({len(story_filters)} filters; {sort_by}; {'search by date' if self.fetcher.search_by_date else 'default search'})"
        )

        comment_filter = CommentFilter(
            min_length=settings.comment_min_length,
            max_length=settings.comment_max_length,
        )

        self.fetcher.story_filter = PipelineFilter(story_filters)
        self.fetcher.comment_filter = comment_filter
        stories = await self.fetcher.fetch_stories(
            max_items=count,
            fetch_comments=True,
            max_comments_per_story=10,
        )

        if sort_by == SortBy.SCORE:
            stories = StorySorter.by_score(stories, reverse=sort_reverse)
        elif sort_by == SortBy.DATE:
            stories = StorySorter.by_date(stories, reverse=sort_reverse)
        elif sort_by == SortBy.ALPHABETICAL:
            stories = StorySorter.by_alphabetical(stories, reverse=sort_reverse)

        stories = stories[:count]

        elapsed = time.time() - start_time
        logger.info(f"Returned {len(stories)} stories in {elapsed:.2f}s")

        return stories
