from hn_sentiment_analysis.logger import get_logger, setup_logging

setup_logging()

logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting Hacker News Sentiment Analysis")
    try:
        logger.debug("Application initialization complete")
        print("Hello from hn-sentiment-analysis!")
        logger.info("Application finished successfully")
    except Exception as e:
        logger.error(f"Application failed: {e}", exc_info=True)
        raise
