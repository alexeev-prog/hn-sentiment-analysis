import asyncio

from hn_sentiment_analysis.hnapi.filters import FetchParams, MinScoreFilter
from hn_sentiment_analysis.pipeline import PipelineParams, build_default_pipeline


async def main():
    pipeline = build_default_pipeline(search_by_date=True)
    result = await pipeline.run(
        PipelineParams(
            fetch=FetchParams(
                count=100000, max_pages=10000, story_filters=[MinScoreFilter(10)]
            )
        )
    )
    for cluster in result.clusters:
        print(f"[{cluster.size:>2} posts] {cluster.display_title}")
    print(f"[{len(result.outliers):>2} posts] unclassified")


if __name__ == "__main__":
    asyncio.run(main())
