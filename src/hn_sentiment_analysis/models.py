from datetime import datetime

from pydantic import BaseModel


class Comment(BaseModel):
    id: int
    text: str
    created_at: str
    author: str | None = None


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

    def build_dict(self) -> dict[str, str]:
        params: dict[str, str] = {}
        if self.query:
            params["query"] = self.query
        if self.tags:
            params["tags"] = ",".join(self.tags)
        if self.numeric_filters:
            params["numericFilters"] = self.numeric_filters
        return params
