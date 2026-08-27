# test.py
import asyncio

from hn_sentiment_analysis.hnapi.fetcher import HNFacade
from hn_sentiment_analysis.hnapi.filters import SortBy


async def main():
    facade = HNFacade(search_by_date=True)
    stories = await facade.get_stories(count=5, sort_by=SortBy.SCORE)

    for story in stories:
        print(f"\n📰 {story.title} (score: {story.score})")
        print(
            f"👤 {story.author} | Comments: {len(story.comments)} | Created: {story.created_at}"
        )
        if story.comments:
            print(f"💬 First comment: {story.comments[0].text[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
