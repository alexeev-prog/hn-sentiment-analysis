from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Mapping, Sequence

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(raw: str) -> str:
    text = html.unescape(raw or "")
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def truncate(text: str, limit: int, *, suffix: str = "…") -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + suffix


_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "i",
        "me",
        "my",
        "what",
        "which",
        "who",
        "whom",
        "new",
        "using",
        "use",
        "used",
        "get",
        "got",
        "one",
        "also",
        "may",
        "might",
        "could",
        "would",
        "show",
        "ask",
        "asked",
        "tell",
        "hn",
    }
)
_TOKEN_RE = re.compile(r"[a-z][a-z0-9+#.\-]*")


def title_tokens(title: str) -> list[str]:
    tokens = []
    for raw in _TOKEN_RE.findall(title.lower()):
        token = raw.strip(".-")
        if len(token) >= 2 and token not in _STOPWORDS:
            tokens.append(token)
    return tokens


def extract_top_terms(
    titles: Sequence[str], corpus_freq: Mapping[str, int], limit: int = 6
) -> list[str]:
    counts: Counter[str] = Counter()
    for title in titles:
        counts.update(title_tokens(title))

    min_count = 2 if len(titles) >= 8 else 1
    scored = [
        (count / (corpus_freq.get(token, 0) + 5), count, token)
        for token, count in counts.items()
        if count >= min_count
    ]
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [token for _, _, token in scored[:limit]]
