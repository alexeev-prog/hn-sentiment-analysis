from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from hn_sentiment_analysis.logger import get_logger
from hn_sentiment_analysis.models import Comment, PipelineResult, Story, StoryCluster

logger = get_logger(__name__)


class JSONSerializer:
    @staticmethod
    def serialize_story(
        story: Story, include_embedding: bool = False
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": story.id,
            "title": story.title,
            "url": story.url,
            "author": story.author,
            "score": story.score,
            "created_at": story.created_at.isoformat() if story.created_at else None,
            "hn_url": story.hn_url,
            "cluster_label": story.cluster_label,
            "comments": [JSONSerializer.serialize_comment(c) for c in story.comments],
        }
        if include_embedding and story.embedding is not None:
            data["embedding"] = story.embedding
        if story.pos_x is not None:
            data["pos_x"] = story.pos_x
            data["pos_y"] = story.pos_y
        return data

    @staticmethod
    def serialize_comment(comment: Comment) -> dict[str, Any]:
        return {
            "id": comment.id,
            "text": comment.text,
            "clean_text": comment.clean_text,
            "created_at": comment.created_at,
            "author": comment.author,
            "story_id": comment.story_id,
        }

    @staticmethod
    def serialize_cluster(cluster: StoryCluster) -> dict[str, Any]:
        return {
            "label": cluster.label,
            "size": cluster.size,
            "total_score": cluster.total_score,
            "total_comments": cluster.total_comments,
            "avg_score": cluster.avg_score,
            "display_title": cluster.display_title,
            "summary": (
                {
                    "title": cluster.summary.title if cluster.summary else None,
                    "description": (
                        cluster.summary.description if cluster.summary else None
                    ),
                    "sentiment": cluster.summary.sentiment if cluster.summary else None,
                    "model": cluster.summary.model if cluster.summary else None,
                }
                if cluster.summary
                else None
            ),
            "stories": [JSONSerializer.serialize_story(s) for s in cluster.stories],
        }

    @staticmethod
    def serialize_result(
        result: PipelineResult, include_embedding: bool = False
    ) -> dict[str, Any]:
        return {
            "metadata": {
                "total_stories": result.total_stories,
                "elapsed_seconds": result.elapsed_seconds,
                "clusters_count": len(result.clusters),
                "outliers_count": len(result.outliers),
                "generated_at": datetime.now().isoformat(),
            },
            "clusters": [JSONSerializer.serialize_cluster(c) for c in result.clusters],
            "outliers": [
                JSONSerializer.serialize_story(s, include_embedding)
                for s in result.outliers
            ],
        }


class JSONReportBuilder:
    def __init__(
        self,
        output_path: str | Path | None = None,
        include_embedding: bool = False,
        indent: int = 2,
    ) -> None:
        self._output = Path(output_path or "hn_clusters.json")
        self._include_embedding = include_embedding
        self._indent = indent

    def build(self, result: PipelineResult) -> Path:
        data = JSONSerializer.serialize_result(result, self._include_embedding)
        self._output.write_text(
            json.dumps(data, indent=self._indent, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"JSON report saved: {self._output.resolve()}")
        return self._output
