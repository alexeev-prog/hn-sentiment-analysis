from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from hn_sentiment_analysis.config import settings
from hn_sentiment_analysis.utils import strip_html

_MAX_EMBEDDING_TEXT_CHARS = 8192
_PARAM_ALIASES = {"numeric_filters": "numericFilters"}


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
    def embedding_text(self) -> str:
        parts = [self.title] * settings.embedding_title_repeats
        for comment in self.comments[: settings.embedding_max_comments]:
            snippet = comment.clean_text[: settings.embedding_comment_chars]
            if snippet:
                parts.append(snippet)
        return "\n".join(parts)[:_MAX_EMBEDDING_TEXT_CHARS]


class ClusterSummary(BaseModel):
    title: str
    description: str
    sentiment: str | None = None
    trend: str | None = None
    model: str | None = None


class ClusterMetrics(BaseModel):
    unique_authors: int = 0
    median_score: float = 0.0
    top_story_score: int = 0
    span_days: int = 0
    avg_age_days: float = 0.0
    posts_per_day: float = 0.0
    recent_share: float = 0.0
    trend_slope: float = 0.0
    trend: str = "stable"
    momentum: int = 0
    lexicon_sentiment: str = "neutral"
    lexicon_score: float = 0.0
    keywords: list[str] = Field(default_factory=list)
    top_domains: list[tuple[str, int]] = Field(default_factory=list)


class StoryCluster(BaseModel):
    label: int
    stories: list[Story] = Field(default_factory=list)
    summary: ClusterSummary | None = None
    metrics: ClusterMetrics | None = None

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
