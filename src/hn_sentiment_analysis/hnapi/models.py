from dataclasses import dataclass
from datetime import datetime, timezone

from hn_sentiment_analysis.logger import get_logger
from hn_sentiment_analysis.models import Story

logger = get_logger(__name__)


@dataclass
class DateRange:
    start_date: datetime | None = None
    end_date: datetime | None = None

    def __post_init__(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date")
        if self.start_date:
            self.start_date = self.start_date.replace(tzinfo=timezone.utc)
            logger.debug(f"Normalized start_date: {self.start_date}")
        if self.end_date:
            self.end_date = self.end_date.replace(tzinfo=timezone.utc)
            logger.debug(f"Normalized end_date: {self.end_date}")


class StoryFilter:
    @staticmethod
    def by_date_range(stories: list[Story], date_range: DateRange) -> list[Story]:
        if not date_range.start_date and not date_range.end_date:
            logger.debug("No date range filters applied")
            return stories

        original_count = len(stories)
        filtered = stories

        if date_range.start_date:
            filtered = [
                story
                for story in filtered
                if story.created_at and story.created_at >= date_range.start_date
            ]
            logger.debug(f"Filtered by start_date: {len(filtered)} remaining")

        if date_range.end_date:
            filtered = [
                story
                for story in filtered
                if story.created_at and story.created_at <= date_range.end_date
            ]
            logger.debug(f"Filtered by end_date: {len(filtered)} remaining")

        logger.info(f"Date range filter: {original_count} -> {len(filtered)} stories")
        return filtered

    @staticmethod
    def by_min_score(stories: list[Story], min_score: int) -> list[Story]:
        if min_score <= 0:
            return stories

        original_count = len(stories)
        filtered = [story for story in stories if story.score >= min_score]
        logger.info(
            f"Score filter (min={min_score}): {original_count} -> {len(filtered)} stories"
        )
        return filtered

    @staticmethod
    def by_author(stories: list[Story], author: str) -> list[Story]:
        if not author:
            return stories

        original_count = len(stories)
        filtered = [story for story in stories if story.author == author]
        logger.info(
            f"Author filter '{author}': {original_count} -> {len(filtered)} stories"
        )
        return filtered

    @staticmethod
    def by_keyword(stories: list[Story], keyword: str) -> list[Story]:
        if not keyword:
            return stories

        original_count = len(stories)
        filtered = [
            story for story in stories if keyword.lower() in story.title.lower()
        ]
        logger.info(
            f"Keyword filter '{keyword}': {original_count} -> {len(filtered)} stories"
        )
        return filtered

    @staticmethod
    def by_tag(stories: list[Story], tag: str) -> list[Story]:
        if not tag:
            return stories

        original_count = len(stories)
        filtered = [story for story in stories if tag in story.tags]
        logger.info(f"Tag filter '{tag}': {original_count} -> {len(filtered)} stories")
        return filtered


class StorySorter:
    @staticmethod
    def by_score(stories: list[Story], reverse: bool = True) -> list[Story]:
        original_count = len(stories)
        sorted_stories = sorted(stories, key=lambda s: s.score or 0, reverse=reverse)
        logger.debug(f"Sorted {original_count} stories by score (reverse={reverse})")
        return sorted_stories

    @staticmethod
    def by_date(stories: list[Story], reverse: bool = True) -> list[Story]:
        original_count = len(stories)
        sorted_stories = sorted(
            stories,
            key=lambda s: s.created_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=reverse,
        )
        logger.debug(f"Sorted {original_count} stories by date (reverse={reverse})")
        return sorted_stories

    @staticmethod
    def by_alphabetical(stories: list[Story], reverse: bool = False) -> list[Story]:
        original_count = len(stories)
        sorted_stories = sorted(stories, key=lambda s: s.title or "", reverse=reverse)
        logger.debug(
            f"Sorted {original_count} stories alphabetically (reverse={reverse})"
        )
        return sorted_stories
