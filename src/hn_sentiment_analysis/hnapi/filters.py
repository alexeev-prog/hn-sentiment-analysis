from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from hn_sentiment_analysis.logger import get_logger
from hn_sentiment_analysis.models import Comment, Story

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


class DateRangeFilter:
    def __init__(self, date_range: DateRange):
        self.date_range = date_range

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


class MinScoreFilter:
    def __init__(self, min_score: int):
        self.min_score = min_score

    def apply(self, stories: list[Story]) -> list[Story]:
        if self.min_score <= 0:
            return stories

        filtered = [story for story in stories if story.score >= self.min_score]
        logger.debug(f"Score filter: {len(stories)} -> {len(filtered)}")
        return filtered


class AuthorFilter:
    def __init__(self, author: str):
        self.author = author

    def apply(self, stories: list[Story]) -> list[Story]:
        if not self.author:
            return stories

        filtered = [story for story in stories if story.author == self.author]
        logger.debug(f"Author filter: {len(stories)} -> {len(filtered)}")
        return filtered


class KeywordFilter:
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


class PipelineFilter:
    def __init__(self, filters: list[StoryFilter]):
        self.filters = filters

    def apply(self, stories: list[Story]) -> list[Story]:
        result = stories
        for filter_obj in self.filters:
            result = filter_obj.apply(result)
        return result
