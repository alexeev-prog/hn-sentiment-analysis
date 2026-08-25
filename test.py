import asyncio
from pprint import pprint

from hn_sentiment_analysis.hnapi.fetcher import HNFetcher


async def main():
    fetcher = HNFetcher()
    stories = await fetcher.fetch_stories(page_limit=1, hits_per_page=10)
    pprint(stories)


if __name__ == "__main__":
    asyncio.run(main())
