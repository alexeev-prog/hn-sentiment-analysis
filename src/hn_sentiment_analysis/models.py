from datetime import datetime
from typing import Any

from pydantic import BaseModel


class Comment(BaseModel):
    id: int
    text: str
    created_at: str
    author: str | None = None
    story_id: int | None = None

    @property
    def length(self) -> int:
        return len(self.text) if self.text else 0


class Story(BaseModel):
    id: int
    title: str
    url: str | None
    created_at: datetime | None
    author: str
    score: int
    tags: list[str] = []
    comments: list[Comment] = []
    embedding: list[float] | None = None
    cluster_label: int | None = None


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
            if value is not None:
                if isinstance(value, list):
                    params[field] = ",".join(value)
                else:
                    params[field] = value
        return params
