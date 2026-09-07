# cli.py
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click

from hn_sentiment_analysis.config import settings
from hn_sentiment_analysis.hnapi.filters import (
    DateRange,
    DateRangeFilter,
    FetchParams,
    KeywordFilter,
    MinScoreFilter,
)
from hn_sentiment_analysis.logger import get_logger, setup_logging
from hn_sentiment_analysis.models import PipelineResult

setup_logging()

logger = get_logger(__name__)


_VERSION = "0.1.0"
_SEPARATOR = "=" * 62


def _build_fetch_params(
    count: int,
    min_score: int,
    days: int | None,
    keyword: str | None,
    fetch_comments: bool,
) -> FetchParams:
    story_filters = []
    if min_score > 0:
        story_filters.append(MinScoreFilter(min_score))
    if days:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        story_filters.append(
            DateRangeFilter(DateRange(start_date=since))  # types: ignore
        )
    if keyword:
        story_filters.append(KeywordFilter(keyword))  # types: ignore
    return FetchParams(
        count=count,
        story_filters=story_filters,  # types: ignore
        fetch_comments=fetch_comments,
    )


def _print_summary(result: PipelineResult, top: int) -> None:
    total = result.total_stories or 1
    clustered = result.total_stories - len(result.outliers)
    noise_pct = 100 * len(result.outliers) / total
    comments = sum(len(story.comments) for story in result.all_stories)

    click.echo(_SEPARATOR)
    click.echo(" Hacker News — AI Cluster Analysis")
    click.echo(_SEPARATOR)
    click.echo(
        f" stories: {result.total_stories} · clustered: {clustered} · "
        f"unclassified: {len(result.outliers)} ({noise_pct:.0f}%)"
    )
    click.echo(
        f" clusters: {len(result.clusters)} · comments: {comments} · "
        f"elapsed: {result.elapsed_seconds / 60:.1f} min"
    )
    if not result.clusters:
        click.echo(_SEPARATOR)
        return

    click.echo("\n top clusters:")
    for rank, cluster in enumerate(result.clusters[:top], start=1):
        badges = ""
        if cluster.summary:
            if cluster.summary.sentiment:
                badges += f"[{cluster.summary.sentiment}]"
            if cluster.summary.momentum:  # types: ignore
                badges += f"[{cluster.summary.momentum}]"  # types: ignore
        click.echo(
            f"  #{rank:<3}{cluster.size:>5} posts {cluster.total_score:>7} pts "
            f"{badges:<22} {cluster.display_title}"
        )
        best = sorted(cluster.stories, key=lambda s: s.score or 0, reverse=True)[:2]
        for story in best:
            click.echo(f"        · {story.title} ({story.score} pts)")
    click.echo(_SEPARATOR)


async def _run_async(
    fetch: FetchParams,
    build_report: bool,
    search_by_date: bool,
    report_path: str | None,
    top: int,
) -> PipelineResult:
    from hn_sentiment_analysis.pipeline import PipelineParams, build_default_pipeline

    pipeline = build_default_pipeline(
        search_by_date=search_by_date, report_path=report_path
    )
    result = await pipeline.run(PipelineParams(fetch=fetch, build_report=build_report))
    _print_summary(result, top)
    if build_report:
        report = Path(report_path or settings.output_html).resolve()
        click.echo(f" report: {report}")
    return result


@click.command()
@click.version_option(version=_VERSION)
@click.option(
    "--count",
    type=int,
    default=None,
    help="Target stories after filtering (default: settings.hn_story_count).",
)
@click.option(
    "--min-score", type=int, default=0, help="Minimum story score (server-side)."
)
@click.option(
    "--days", type=int, default=None, help="Only stories from the last N days."
)
@click.option("--keyword", default=None, help="Substring filter for titles.")
@click.option(
    "--relevance", is_flag=True, help="Relevance search instead of date walk."
)
@click.option("--no-comments", is_flag=True, help="Skip fetching comments.")
@click.option("--no-report", is_flag=True, help="Skip HTML report generation.")
@click.option("--top", type=int, default=15, help="Clusters to show in summary.")
@click.option("--out", type=click.Path(), default=None, help="HTML report path.")
def run(
    count: int | None,
    min_score: int,
    days: int | None,
    keyword: str | None,
    relevance: bool,
    no_comments: bool,
    no_report: bool,
    top: int,
    out: str | None,
) -> None:
    """Run the Hacker News AI cluster analysis pipeline."""
    setup_logging()
    fetch = _build_fetch_params(
        count=count if count is not None else settings.hn_story_count,
        min_score=min_score,
        days=days,
        keyword=keyword,
        fetch_comments=not no_comments,
    )
    asyncio.run(
        _run_async(
            fetch=fetch,
            build_report=not no_report,
            search_by_date=not relevance,
            report_path=out,
            top=top,
        )
    )


def main():
    run()
