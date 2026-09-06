from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ai_api_key: str | None = None
    ai_base_url: str = "https://api.bothub.ai/v1"
    ai_default_model: str = "gpt-5.6-luna"
    ai_request_timeout: float = 60.0
    ai_max_concurrent: int = 4
    ai_max_retries: int = 2
    ai_batch_size: int = 8
    ai_retry_backoff: float = 2.0
    ai_temperature: float = 0.2
    ai_max_tokens: int = 300
    ai_max_stories_in_prompt: int = 15
    ai_max_comments_in_prompt: int = 3
    ai_prompt_comment_chars: int = 200
    ai_summary_language: str = "English"

    hn_story_count: int = 300
    hn_hits_per_page: int = 100
    hn_max_pages: int = 50
    hn_comments_per_story: int = 10
    hn_request_concurrency: int = 8
    hn_request_delay: float = 0.1
    hn_request_timeout: float = 20.0
    hn_request_retries: int = 3
    hn_retry_backoff: float = 1.0
    hn_failure_threshold: int = 10
    comment_min_length: int = 32
    comment_max_length: int = 8192

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    embedding_title_repeats: int = 2
    embedding_max_comments: int = 5
    embedding_comment_chars: int = 256

    cluster_min_size: int = 5
    cluster_auto_min_size: bool = True
    cluster_min_samples: int = 3
    cluster_assign_outliers: bool = True
    cluster_outlier_threshold: float = 0.25
    cluster_trim_outliers: bool = True
    cluster_trim_z: float = 2.5
    cluster_selection_method: str = "leaf"
    umap_n_neighbors: int = 15
    umap_n_components: int = 10
    umap_min_dist: float = 0.0

    metrics_recent_days: int = 3
    metrics_max_keywords: int = 6
    metrics_max_domains: int = 4

    output_html: str = "hn_clusters.html"
    report_charts: bool = True
    report_comments_per_story: int = 3
    report_comment_chars: int = 240

    log_level: str = "INFO"
    log_format: str = "plain"
    log_file: str = "hn_sentiment.log"
    log_max_bytes: int = 10485760
    log_backup_count: int = 5
    log_console: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
