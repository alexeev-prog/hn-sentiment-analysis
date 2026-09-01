from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from hn_sentiment_analysis.config import settings
from hn_sentiment_analysis.logger import get_logger
from hn_sentiment_analysis.models import Comment, QueryParams, Story

logger = get_logger(__name__)


class SortBy(str, Enum):
    SCORE = "score"
    DATE = "date"
    ALPHABETICAL = "alphabetical"


@dataclass
class DateRange:
    start_date: datetime | None = None
    end_date: datetime | None = None

    def __post_init__(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date")
        if self.start_date:
            self.start_date = self.start_date.replace(tzinfo=timezone.utc)
        if self.end_date:
            self.end_date = self.end_date.replace(tzinfo=timezone.utc)


class StoryFilter(Protocol):
    def apply(self, stories: list[Story]) -> list[Story]: ...


class AlgoliaStoryFilter(ABC):
    @abstractmethod
    def apply(self, stories: list[Story]) -> list[Story]: ...

    def numeric_filters(self) -> list[str]:
        return []

    def query_tags(self) -> list[str]:
        return []


def algolia_constraints(
    story_filters: Sequence[StoryFilter],
) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    tags: list[str] = []
    for story_filter in story_filters:
        if isinstance(story_filter, AlgoliaStoryFilter):
            numeric.extend(story_filter.numeric_filters())
            tags.extend(story_filter.query_tags())
    return numeric, tags


class DateRangeFilter(AlgoliaStoryFilter):
    def __init__(self, date_range: DateRange):
        self.date_range = date_range

    def numeric_filters(self) -> list[str]:
        constraints: list[str] = []
        start = self.date_range.start_date
        end = self.date_range.end_date
        if start:
            constraints.append(f"created_at_i>{int(start.timestamp())}")
        if end:
            constraints.append(f"created_at_i<{int(end.timestamp())}")
        return constraints

    def apply(self, stories: list[Story]) -> list[Story]:
        if not self.date_range.start_date and not self.date_range.end_date:
            return stories

        filtered = stories

        if self.date_range.start_date:
            filtered = [
                story
                for story in filtered
                if story.created_at and story.created_at >= self.date_range.start_date
            ]

        if self.date_range.end_date:
            filtered = [
                story
                for story in filtered
                if story.created_at and story.created_at <= self.date_range.end_date
            ]

        logger.debug(f"Date range filter: {len(stories)} -> {len(filtered)}")
        return filtered


class MinScoreFilter(AlgoliaStoryFilter):
    def __init__(self, min_score: int):
        self.min_score = min_score

    def numeric_filters(self) -> list[str]:
        return [f"points>={self.min_score}"] if self.min_score > 0 else []

    def apply(self, stories: list[Story]) -> list[Story]:
        if self.min_score <= 0:
            return stories

        filtered = [story for story in stories if story.score >= self.min_score]
        logger.debug(f"Score filter: {len(stories)} -> {len(filtered)}")
        return filtered


class AuthorFilter(AlgoliaStoryFilter):
    def __init__(self, author: str):
        self.author = author

    def query_tags(self) -> list[str]:
        return [f"author_{self.author}"] if self.author else []

    def apply(self, stories: list[Story]) -> list[Story]:
        if not self.author:
            return stories

        filtered = [story for story in stories if story.author == self.author]
        logger.debug(f"Author filter: {len(stories)} -> {len(filtered)}")
        return filtered


class KeywordFilter(AlgoliaStoryFilter):
    def __init__(self, keyword: str):
        self.keyword = keyword.lower()

    def apply(self, stories: list[Story]) -> list[Story]:
        if not self.keyword:
            return stories

        filtered = [story for story in stories if self.keyword in story.title.lower()]
        logger.debug(f"Keyword filter: {len(stories)} -> {len(filtered)}")
        return filtered


class CommentFilter:
    def __init__(self, min_length: int | None = None, max_length: int | None = None):
        self.min_length = min_length or 0
        self.max_length = max_length or float("inf")

    def apply_to_stories(self, stories: list[Story]) -> list[Story]:
        for story in stories:
            story.comments = self.apply_to_comments(story.comments)
        return stories

    def apply_to_comments(self, comments: list[Comment]) -> list[Comment]:
        filtered = [
            comment
            for comment in comments
            if self.min_length <= comment.length <= self.max_length
        ]
        logger.debug(f"Comment filter: {len(comments)} -> {len(filtered)}")
        return filtered


class StorySorter:
    @staticmethod
    def by_score(stories: list[Story], reverse: bool = True) -> list[Story]:
        return sorted(stories, key=lambda s: s.score or 0, reverse=reverse)

    @staticmethod
    def by_date(stories: list[Story], reverse: bool = True) -> list[Story]:
        return sorted(
            stories,
            key=lambda s: s.created_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=reverse,
        )

    @staticmethod
    def by_alphabetical(stories: list[Story], reverse: bool = False) -> list[Story]:
        return sorted(stories, key=lambda s: s.title or "", reverse=reverse)


_SORTERS: dict[SortBy, Callable[[list[Story], bool], list[Story]]] = {
    SortBy.SCORE: StorySorter.by_score,
    SortBy.DATE: StorySorter.by_date,
    SortBy.ALPHABETICAL: StorySorter.by_alphabetical,
}


def sort_stories(
    stories: list[Story], sort_by: SortBy | str, reverse: bool = True
) -> list[Story]:
    return _SORTERS[SortBy(sort_by)](stories, reverse)


class PipelineFilter:
    def __init__(self, filters: list[StoryFilter]):
        self.filters = filters

    def apply(self, stories: list[Story]) -> list[Story]:
        result = stories
        for filter_obj in self.filters:
            result = filter_obj.apply(result)
        return result


@dataclass(slots=True)
class FetchParams:
    count: int = settings.hn_story_count
    story_filters: list[StoryFilter] = field(default_factory=list)
    comment_filter: CommentFilter | None = None
    sort_by: SortBy | str = SortBy.SCORE
    sort_reverse: bool = True
    query: QueryParams | None = None
    hits_per_page: int = settings.hn_hits_per_page
    fetch_comments: bool = True
    max_comments_per_story: int = settings.hn_comments_per_story
    max_pages: int = settings.hn_max_pages
