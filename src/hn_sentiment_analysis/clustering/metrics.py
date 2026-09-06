from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Sequence
from urllib.parse import urlparse

from hn_sentiment_analysis.config import settings
from hn_sentiment_analysis.models import ClusterMetrics, Story, StoryCluster

_WORD_RE = re.compile(r"[a-z']+")
_TREND_RATIO = 0.15
_MIN_SIGNAL_WORDS = 5

_POSITIVE = {
    "love": 2,
    "great": 2,
    "best": 2,
    "awesome": 2,
    "excellent": 2,
    "amazing": 2,
    "fantastic": 2,
    "brilliant": 2,
    "perfect": 2,
    "wonderful": 2,
    "incredible": 2,
    "outstanding": 2,
    "impressed": 2,
    "impressive": 2,
    "delightful": 2,
    "superb": 2,
    "good": 1,
    "nice": 1,
    "useful": 1,
    "helpful": 1,
    "solid": 1,
    "fast": 1,
    "elegant": 1,
    "clean": 1,
    "works": 1,
    "working": 1,
    "recommend": 1,
    "recommended": 1,
    "thanks": 1,
    "cool": 1,
    "interesting": 1,
    "promising": 1,
    "excited": 1,
    "exciting": 1,
    "enjoy": 1,
    "enjoyed": 1,
    "reliable": 1,
    "secure": 1,
    "powerful": 1,
    "improved": 1,
    "better": 1,
    "win": 1,
    "wins": 1,
    "progress": 1,
    "success": 1,
    "successful": 1,
    "praise": 1,
    "beautiful": 1,
    "easy": 1,
    "intuitive": 1,
    "robust": 1,
    "stable": 1,
    "valuable": 1,
    "boost": 1,
    "breakthrough": 1,
    "gem": 1,
    "underrated": 1,
}

_NEGATIVE = {
    "hate": 2,
    "worst": 2,
    "terrible": 2,
    "awful": 2,
    "horrible": 2,
    "disaster": 2,
    "disastrous": 2,
    "broken": 2,
    "useless": 2,
    "garbage": 2,
    "trash": 2,
    "scam": 2,
    "fraud": 2,
    "nightmare": 2,
    "pathetic": 2,
    "unacceptable": 2,
    "atrocious": 2,
    "appalling": 2,
    "dreadful": 2,
    "bad": 1,
    "poor": 1,
    "slow": 1,
    "buggy": 1,
    "flawed": 1,
    "frustrating": 1,
    "frustrated": 1,
    "annoying": 1,
    "disappointed": 1,
    "disappointing": 1,
    "overpriced": 1,
    "complicated": 1,
    "confusing": 1,
    "worse": 1,
    "fail": 1,
    "fails": 1,
    "failed": 1,
    "failing": 1,
    "failure": 1,
    "crash": 1,
    "crashes": 1,
    "crashed": 1,
    "leak": 1,
    "leaked": 1,
    "breach": 1,
    "hacked": 1,
    "lawsuit": 1,
    "banned": 1,
    "regression": 1,
    "decline": 1,
    "declining": 1,
    "misleading": 1,
    "overhyped": 1,
    "pointless": 1,
    "waste": 1,
    "mediocre": 1,
    "unstable": 1,
    "insecure": 1,
    "vulnerability": 1,
    "exploit": 1,
    "layoffs": 1,
    "toxic": 1,
    "dead": 1,
    "died": 1,
    "kills": 1,
    "killed": 1,
}

_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "them",
    "his",
    "her",
    "their",
    "our",
    "your",
    "my",
    "me",
    "us",
    "not",
    "no",
    "yes",
    "so",
    "do",
    "does",
    "did",
    "done",
    "have",
    "has",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "must",
    "than",
    "more",
    "most",
    "some",
    "any",
    "all",
    "one",
    "two",
    "also",
    "just",
    "about",
    "into",
    "over",
    "under",
    "out",
    "up",
    "down",
    "after",
    "before",
    "when",
    "where",
    "why",
    "how",
    "what",
    "which",
    "who",
    "whom",
    "there",
    "here",
    "now",
    "new",
    "get",
    "got",
    "make",
    "made",
    "use",
    "used",
    "using",
    "like",
    "well",
    "only",
    "very",
    "much",
    "many",
    "few",
    "own",
    "same",
    "hn",
    "comment",
    "comments",
    "says",
    "said",
    "story",
    "post",
    "posts",
    "points",
    "hours",
    "ago",
    "day",
    "days",
    "year",
    "years",
    "time",
    "people",
    "really",
    "thing",
    "things",
    "way",
    "ways",
    "lot",
    "bit",
    "long",
    "short",
    "old",
    "first",
    "last",
    "top",
    "via",
    "com",
    "www",
    "http",
    "https",
    "amp",
    "don",
    "re",
    "ve",
    "ll",
    "y",
}


def analyze_clusters(clusters: list[StoryCluster], now: datetime | None = None) -> None:
    moment = now or datetime.now(timezone.utc)
    for cluster in clusters:
        cluster.metrics = _compute_metrics(cluster.stories, moment)


def _compute_metrics(stories: list[Story], now: datetime) -> ClusterMetrics:
    dates = [_aware(story.created_at) for story in stories if story.created_at]
    ages = sorted((now - moment).days for moment in dates)
    scores = sorted(story.score or 0 for story in stories)

    recent_days = settings.metrics_recent_days
    recent_share = (
        sum(1 for age in ages if age < recent_days) / len(ages) if ages else 0.0
    )
    span_days = (ages[-1] - ages[0] + 1) if ages else 0
    slope, trend = _trend(ages)
    momentum = round(100 * (0.5 * recent_share + 0.5 * max(0.0, min(1.0, slope))))
    lexicon_label, lexicon_score = _lexicon_sentiment(stories)

    return ClusterMetrics(
        unique_authors=len({story.author for story in stories}),
        median_score=float(scores[len(scores) // 2]) if scores else 0.0,
        top_story_score=scores[-1] if scores else 0,
        span_days=span_days,
        avg_age_days=sum(ages) / len(ages) if ages else 0.0,
        posts_per_day=len(ages) / span_days if span_days else 0.0,
        recent_share=round(recent_share, 3),
        trend_slope=round(slope, 4),
        trend=trend,
        momentum=momentum,
        lexicon_sentiment=lexicon_label,
        lexicon_score=lexicon_score,
        keywords=_keywords(stories),
        top_domains=_top_domains(stories),
    )


def _trend(ages: Sequence[int]) -> tuple[float, str]:
    if len(ages) < 3:
        return 0.0, "stable"

    counts = Counter(ages)
    origin = ages[-1]
    x = [origin - age for age in sorted(counts)]
    y = [counts[origin - day] for day in x]
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator == 0:
        return 0.0, "stable"

    slope = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / denominator
    normalized = slope / (mean_y or 1.0)
    if normalized > _TREND_RATIO:
        label = "rising"
    elif normalized < -_TREND_RATIO:
        label = "fading"
    else:
        label = "stable"
    return normalized, label


def _lexicon_sentiment(stories: Sequence[Story]) -> tuple[str, float]:
    positive = 0
    negative = 0
    total_words = 0
    for story in stories:
        for comment in story.comments:
            words = _WORD_RE.findall(comment.clean_text.lower())
            total_words += len(words)
            for word in words:
                positive += _POSITIVE.get(word, 0)
                negative += _NEGATIVE.get(word, 0)

    if positive + negative < _MIN_SIGNAL_WORDS:
        return "neutral", 0.0

    score = round((positive - negative) / total_words, 4) if total_words else 0.0
    ratio = (positive - negative) / (positive + negative)
    if ratio > 0.2:
        label = "positive"
    elif ratio < -0.2:
        label = "negative"
    else:
        label = "mixed"
    return label, score


def _keywords(stories: Sequence[Story]) -> list[str]:
    unigrams: Counter[str] = Counter()
    bigrams: Counter[str] = Counter()
    for story in stories:
        words = [
            word
            for word in _WORD_RE.findall(story.title.lower())
            if len(word) > 2 and word not in _STOPWORDS
        ]
        unigrams.update(words)
        bigrams.update(f"{a} {b}" for a, b in zip(words, words[1:]))

    candidates = [
        (phrase, count) for phrase, count in bigrams.items() if count >= 3
    ] + [(phrase, count) for phrase, count in unigrams.items() if count >= 2]
    candidates.sort(key=lambda item: item[1], reverse=True)

    seen: set[str] = set()
    keywords = []
    for phrase, _ in candidates:
        parts = set(phrase.split())
        if parts & seen:
            continue
        seen.update(parts)
        keywords.append(phrase)
        if len(keywords) >= settings.metrics_max_keywords:
            break
    return keywords


def _top_domains(stories: Sequence[Story]) -> list[tuple[str, int]]:
    domains = Counter(
        domain for domain in (_domain(story.url) for story in stories) if domain
    )
    threshold = max(2, len(stories) // 25)
    return [
        (domain, count)
        for domain, count in domains.most_common(settings.metrics_max_domains)
        if count >= threshold
    ]


def _domain(url: str | None) -> str | None:
    if not url:
        return None
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.") or None


def _aware(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment
