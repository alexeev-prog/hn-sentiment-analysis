# models.py
from datetime import datetime, timezone
from statistics import median
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from hn_sentiment_analysis.config import settings
from hn_sentiment_analysis.utils import strip_html

_MAX_EMBEDDING_TEXT_CHARS = 8192
_PARAM_ALIASES = {"numeric_filters": "numericFilters"}
_FRESH_HOURS = 48.0


def _median_of(values: list[float | None]) -> float:
    present = [value for value in values if value is not None]
    return float(median(present)) if present else 0.0


class Comment(BaseModel):
    id: int
    text: str
    created_at: str
    author: str | None = None
    story_id: int | None = None

    @property
    def clean_text(self) -> str:
        return strip_html(self.text)

    @property
    def length(self) -> int:
        return len(self.clean_text)


class Story(BaseModel):
    id: int
    title: str
    url: str | None
    created_at: datetime | None
    author: str
    score: int
    tags: list[str] = Field(default_factory=list)
    comments: list[Comment] = Field(default_factory=list)
    embedding: list[float] | None = None
    cluster_label: int | None = None
    pos_x: float | None = None
    pos_y: float | None = None

    @property
    def hn_url(self) -> str:
        return f"https://news.ycombinator.com/item?id={self.id}"

    @property
    def domain(self) -> str | None:
        if not self.url:
            return None
        host = urlsplit(self.url).netloc.removeprefix("www.")
        return host or None

    @property
    def age_hours(self) -> float | None:
        if self.created_at is None:
            return None
        elapsed = datetime.now(timezone.utc) - self.created_at
        return max(0.0, elapsed.total_seconds() / 3600)

    @property
    def points_per_hour(self) -> float | None:
        age = self.age_hours
        if age is None:
            return None
        return (self.score or 0) / max(1.0, age)

    @property
    def embedding_text(self) -> str:
        if settings.embedding_include_domain and self.domain:
            header = f"{self.domain} — {self.title}"
        else:
            header = self.title
        parts = [header] * settings.embedding_title_repeats
        for comment in self.comments[: settings.embedding_max_comments]:
            snippet = comment.clean_text[: settings.embedding_comment_chars]
            if snippet:
                parts.append(snippet)
        return "\n".join(parts)[:_MAX_EMBEDDING_TEXT_CHARS]


class ClusterSummary(BaseModel):
    title: str
    description: str
    sentiment: str | None = None
    momentum: str | None = None
    model: str | None = None


class StoryCluster(BaseModel):
    label: int
    stories: list[Story] = Field(default_factory=list)
    summary: ClusterSummary | None = None
    top_terms: list[str] = Field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.stories)

    @property
    def total_score(self) -> int:
        return sum(story.score or 0 for story in self.stories)

    @property
    def total_comments(self) -> int:
        return sum(len(story.comments) for story in self.stories)

    @property
    def avg_score(self) -> float:
        return self.total_score / self.size if self.size else 0.0

    @property
    def display_title(self) -> str:
        return self.summary.title if self.summary else f"Cluster #{self.label}"

    @property
    def median_score(self) -> float:
        return _median_of([float(story.score or 0) for story in self.stories])

    @property
    def top_score(self) -> int:
        return max((story.score or 0 for story in self.stories), default=0)

    @property
    def unique_authors(self) -> int:
        return len({story.author for story in self.stories})

    @property
    def velocity(self) -> float:
        return _median_of([story.points_per_hour for story in self.stories])

    @property
    def age_hours(self) -> float:
        return _median_of([story.age_hours for story in self.stories])

    @property
    def fresh_share(self) -> float:
        ages = [
            age
            for age in (story.age_hours for story in self.stories)
            if age is not None
        ]
        if not ages:
            return 0.0
        return sum(1 for age in ages if age <= _FRESH_HOURS) / len(ages)

    @property
    def top_domains(self) -> list[str]:
        counts: dict[str, int] = {}
        for story in self.stories:
            domain = story.domain
            if domain:
                counts[domain] = counts.get(domain, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [domain for domain, _ in ranked[:3]]


class PipelineResult(BaseModel):
    clusters: list[StoryCluster] = Field(default_factory=list)
    outliers: list[Story] = Field(default_factory=list)
    total_stories: int = 0
    elapsed_seconds: float = 0.0

    @property
    def all_stories(self) -> list[Story]:
        return [
            story for cluster in self.clusters for story in cluster.stories
        ] + self.outliers


class QueryParams(BaseModel):
    query: str | None = None
    tags: list[str] | None = None
    numeric_filters: str | None = None
    filters: str | None = None
    hitsPerPage: int = 100
    page: int = 0

    def build_dict(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for field in self.model_fields:
            value = getattr(self, field)
            if value is None:
                continue
            if isinstance(value, list):
                value = ",".join(value)
            params[_PARAM_ALIASES.get(field, field)] = value
        return params
