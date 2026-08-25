from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ai_api_key: str | None = None
    ai_base_url: str = "https://api.bothub.ai/v1"
    ai_default_model: str = "gpt-5.6-luna"

    hn_max_items: int = 500
    hn_comments_per_story: int = 3

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    cluster_min_size: int = 5
    umap_n_neighbors: int = 15

    output_html: str = "hn_clusters.html"

    log_level: str = "INFO"
    log_format: str = "plain"
    log_file: str = "hn_sentiment.log"
    log_max_bytes: int = 10485760
    log_backup_count: int = 5
    log_console: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
