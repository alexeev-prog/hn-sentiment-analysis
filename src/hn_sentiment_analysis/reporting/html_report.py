# reporting/html_report.py
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from hn_sentiment_analysis.config import settings
from hn_sentiment_analysis.logger import get_logger
from hn_sentiment_analysis.models import Comment, PipelineResult, Story, StoryCluster
from hn_sentiment_analysis.utils import strip_html, truncate

logger = get_logger(__name__)

_CSS = """
:root {
  color-scheme: light dark;
  --bg: #f5f5f3; --card: #ffffff; --fg: #1f2328; --muted: #6e7781;
  --border: #e3e4e1; --accent: #ff6600;
  --accent-soft: rgba(255, 102, 0, 0.1);
  --shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1113; --card: #17191c; --fg: #e6e6e3; --muted: #9aa0a6;
    --border: #26292d; --accent-soft: rgba(255, 102, 0, 0.16);
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
  }
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg); color: var(--fg); line-height: 1.55;
  margin: 0 auto; max-width: 1100px; padding: 1.5rem 1.25rem 4rem;
}
header.page {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.4rem 1.5rem;
  padding-bottom: 1rem; border-bottom: 1px solid var(--border);
}
h1 { margin: 0; font-size: 1.55rem; letter-spacing: -0.02em; }
h1 .accent { color: var(--accent); }
.stats { margin: 0; color: var(--muted); font-size: 0.9rem; }
.kpis {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr));
  gap: 0.75rem; margin: 1.5rem 0 1rem;
}
.kpi {
  background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  padding: 0.7rem 0.85rem; box-shadow: var(--shadow);
}
.kpi-value { display: block; font-size: 1.3rem; font-weight: 700; }
.kpi-label {
  color: var(--muted); font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.07em;
}
h2.section { margin: 2rem 0 0.35rem; font-size: 1.15rem; }
.hint { color: var(--muted); font-size: 0.8rem; margin: 0 0 0.9rem; }
.chart-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  gap: 1.25rem;
}
figure.chart {
  margin: 0; background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 0.9rem; box-shadow: var(--shadow);
  cursor: zoom-in;
}
figure.chart.wide { grid-column: 1 / -1; }
figure.chart figcaption {
  display: flex; align-items: center; gap: 0.4rem; color: var(--muted);
  font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem;
}
figure.chart figcaption::after {
  content: "⤢"; margin-left: auto; opacity: 0.55; font-weight: 400;
}
.chart-body { display: flex; flex-direction: column; }
figure.chart svg { display: block; width: 100%; height: auto; }
.legend {
  list-style: none; display: flex; flex-wrap: wrap; gap: 0.4rem 1rem;
  margin: 0.7rem 0 0; padding: 0;
}
.legend li {
  display: inline-flex; align-items: center; gap: 0.4rem;
  color: var(--muted); font-size: 0.8rem;
}
.toc { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1.5rem 0; }
.toc a {
  display: inline-flex; align-items: center; gap: 0.45rem;
  background: var(--card); border: 1px solid var(--border); border-radius: 999px;
  padding: 0.3rem 0.8rem; color: var(--fg); font-size: 0.82rem;
  text-decoration: none;
}
.toc a:hover { border-color: var(--accent); }
.chip { width: 0.65rem; height: 0.65rem; border-radius: 3px; flex: none; }
.cluster {
  background: var(--card); border: 1px solid var(--border);
  border-left: 4px solid var(--c, var(--accent)); border-radius: 12px;
  box-shadow: var(--shadow); margin: 1.1rem 0; overflow: hidden;
}
.cluster > header {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.3rem 0.8rem;
  padding: 0.8rem 1.1rem 0.6rem;
  background: linear-gradient(90deg, var(--c-bg, var(--accent-soft)), transparent 70%);
}
.cluster h2 { margin: 0; font-size: 1.1rem; }
.rank { color: var(--c, var(--accent)); font-weight: 700; font-size: 0.8rem; }
.sentiment, .momentum {
  font-size: 0.72rem; font-weight: 600; text-transform: capitalize;
  border-radius: 999px; padding: 0.08rem 0.6rem;
}
.sentiment.s-positive { color: #2f9e44; background: rgba(47, 158, 68, 0.15); }
.sentiment.s-negative { color: #e03131; background: rgba(224, 49, 49, 0.15); }
.sentiment.s-mixed { color: #e8590c; background: rgba(232, 89, 12, 0.15); }
.sentiment.s-neutral { color: var(--muted); background: rgba(134, 142, 150, 0.18); }
.momentum.s-rising { color: #2f9e44; background: rgba(47, 158, 68, 0.15); }
.momentum.s-steady { color: var(--muted); background: rgba(134, 142, 150, 0.18); }
.momentum.s-fading { color: #e8590c; background: rgba(232, 89, 12, 0.15); }
.terms {
  list-style: none; display: flex; flex-wrap: wrap; gap: 0.3rem;
  width: 100%; margin: 0.2rem 0 0; padding: 0;
}
.terms .term {
  background: var(--c-bg, var(--accent-soft)); color: var(--muted);
  border-radius: 999px; padding: 0.05rem 0.55rem; font-size: 0.72rem;
}
.meta { color: var(--muted); font-size: 0.83rem; margin: 0; }
.cluster .meta { width: 100%; }
.description {
  margin: 0; padding: 0.8rem 1.1rem; border-bottom: 1px solid var(--border);
}
.story {
  padding: 0.65rem 1.1rem; border-bottom: 1px solid var(--border);
  transition: background 0.15s;
}
.story:last-child { border-bottom: none; }
.story:hover { background: var(--c-bg, var(--accent-soft)); }
.story a.title { color: var(--fg); font-weight: 600; text-decoration: none; }
.story a.title:hover { color: var(--c, var(--accent)); }
.story a.discussion { color: var(--muted); font-size: 0.82rem; text-decoration: none; }
.story a.discussion:hover { color: var(--c, var(--accent)); }
.story .meta { margin: 0.15rem 0 0; }
.story details { margin-top: 0.3rem; }
.story summary { color: var(--muted); font-size: 0.82rem; cursor: pointer; }
.comment {
  border-left: 3px solid var(--c, var(--border)); margin: 0.4rem 0;
  padding: 0.1rem 0 0.1rem 0.7rem; color: var(--muted); font-size: 0.88rem;
}
.comment b { color: var(--fg); }
details.outliers {
  background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  margin-top: 1.75rem; padding: 0.9rem 1.1rem;
}
details.outliers summary { font-weight: 600; cursor: pointer; }
details.outliers ul { margin: 0.75rem 0 0; padding-left: 1.2rem; }
details.outliers li { margin: 0.25rem 0; font-size: 0.92rem; }
footer.page {
  margin-top: 2.5rem; padding-top: 0.75rem; border-top: 1px solid var(--border);
  color: var(--muted); font-size: 0.8rem;
}
.bar, .dot { transition: opacity 0.15s; }
.bar:hover { opacity: 0.7; }
.dot:hover { opacity: 1; stroke: var(--fg); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 11px; }
.val { fill: var(--fg); font-size: 11px; }
.axis { stroke: var(--border); stroke-width: 1; }
.grid { stroke: var(--border); stroke-width: 1; stroke-dasharray: 3 3; }
svg text { font-family: inherit; }
a { color: var(--accent); }
#chart-viewer {
  background: var(--card); color: var(--fg); border: 1px solid var(--border);
  border-radius: 14px; padding: 0; width: min(96vw, 1500px); height: 90vh;
  box-shadow: var(--shadow);
}
#chart-viewer::backdrop { background: rgba(0, 0, 0, 0.62); }
#chart-viewer:fullscreen { width: 100vw; height: 100vh; border: none; border-radius: 0; }
#chart-viewer[open] { display: flex; flex-direction: column; }
.viewer-head {
  display: flex; align-items: center; gap: 0.5rem; flex: none;
  padding: 0.6rem 0.9rem; border-bottom: 1px solid var(--border);
}
.viewer-title {
  margin: 0; font-size: 0.95rem; font-weight: 600; flex: 1; min-width: 0;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.viewer-head button {
  background: var(--bg); color: var(--fg); border: 1px solid var(--border);
  border-radius: 8px; min-width: 30px; height: 30px; padding: 0 0.55rem;
  font-size: 0.85rem; line-height: 1; cursor: pointer;
}
.viewer-head button:hover { border-color: var(--accent); color: var(--accent); }
.viewer-body {
  flex: 1; min-height: 0; display: flex; flex-direction: column;
  padding: 0.75rem 1rem 1rem; overflow: hidden;
}
#chart-viewer svg { flex: 1; min-height: 0; width: 100%; }
#chart-viewer .legend { flex: none; }
"""

_VIEWER_JS = """
(() => {
  const viewer = document.getElementById('chart-viewer');
  const figures = Array.from(document.querySelectorAll('figure.chart'));
  if (!viewer || figures.length === 0) return;
  const body = viewer.querySelector('.viewer-body');
  const title = viewer.querySelector('.viewer-title');
  const fsButton = viewer.querySelector('[data-act="fs"]');
  const exportStyle = '.tick{fill:#6e7781;font-size:11px}' +
    '.val{fill:#1f2328;font-size:11px}' +
    '.axis{stroke:#e3e4e1;stroke-width:1}' +
    '.grid{stroke:#e3e4e1;stroke-width:1;stroke-dasharray:3 3}';
  const exportColors = {
    'var(--accent)': '#ff6600',
    'var(--fg)': '#1f2328',
    'var(--muted)': '#6e7781',
    'var(--border)': '#e3e4e1',
  };
  let index = 0;
  let fileName = 'chart';

  const render = () => {
    const figure = figures[index];
    const caption = figure.querySelector('figcaption').textContent.trim();
    title.textContent = caption;
    body.innerHTML = figure.querySelector('.chart-body').innerHTML;
    fileName = caption.toLowerCase().replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '') || 'chart';
  };
  const show = (i) => {
    index = (i + figures.length) % figures.length;
    render();
    if (!viewer.open) viewer.showModal();
  };
  const toggleFullscreen = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else viewer.requestFullscreen().catch(() => {});
  };
  const serialize = () => {
    const source = body.querySelector('svg');
    if (!source) return null;
    const clone = source.cloneNode(true);
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    const box = (clone.getAttribute('viewBox') || '0 0 720 300')
      .split(/\\s+/).map(Number);
    const width = box[2] || 720;
    const height = box[3] || 300;
    clone.setAttribute('width', width);
    clone.setAttribute('height', height);
    const style = document.createElementNS('http://www.w3.org/2000/svg', 'style');
    style.textContent = exportStyle;
    clone.insertBefore(style, clone.firstChild);
    let markup = new XMLSerializer().serializeToString(clone);
    for (const [name, value] of Object.entries(exportColors)) {
      markup = markup.split(name).join(value);
    }
    return { markup, width, height };
  };
  const saveBlob = (blob, ext) => {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = fileName + '.' + ext;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 5000);
  };
  const savePng = () => {
    const data = serialize();
    if (!data) return;
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = data.width * 2;
      canvas.height = data.height * 2;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => blob && saveBlob(blob, 'png'));
    };
    image.src = 'data:image/svg+xml;charset=utf-8,' +
      encodeURIComponent(data.markup);
  };
  const saveSvg = () => {
    const data = serialize();
    if (data) saveBlob(new Blob([data.markup], { type: 'image/svg+xml' }), 'svg');
  };

  figures.forEach((figure, i) => figure.addEventListener('click', () => show(i)));
  viewer.addEventListener('click', (event) => {
    const action = event.target.closest('button')?.dataset.act;
    if (action === 'close') viewer.close();
    else if (action === 'prev') show(index - 1);
    else if (action === 'next') show(index + 1);
    else if (action === 'fs') toggleFullscreen();
    else if (action === 'svg') saveSvg();
    else if (action === 'png') savePng();
    else if (event.target === viewer) viewer.close();
  });
  viewer.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowLeft') show(index - 1);
    else if (event.key === 'ArrowRight') show(index + 1);
    else if (event.key.toLowerCase() === 'f') toggleFullscreen();
  });
  viewer.addEventListener('close', () => {
    if (document.fullscreenElement) document.exitFullscreen();
  });
  document.addEventListener('fullscreenchange', () => {
    fsButton.textContent = document.fullscreenElement ? '⤡' : '⤢';
  });
})();
"""

_PALETTE = (
    "#ff6600",
    "#1c7ed6",
    "#2f9e44",
    "#9c36b5",
    "#e8590c",
    "#0ca678",
    "#7048e8",
    "#c2255c",
    "#f08c00",
    "#1098ad",
    "#e64980",
    "#5c940d",
)
_NOISE_COLOR = "#868e96"
_SCORE_EDGES = (0, 10, 25, 50, 100, 200, 500)
_SENTIMENTS = ("positive", "negative", "mixed", "neutral")
_SENTIMENT_COLORS = ("#2f9e44", "#e03131", "#f08c00", "#868e96")
_MAX_LEGEND_ITEMS = 12
_MAX_TOC_ITEMS = 20
_MAX_CHART_CLUSTERS = 15
_MAX_TOP_STORIES = 10


@dataclass(slots=True)
class _Analytics:
    stories: list[Story] = field(default_factory=list)
    comments: int = 0
    points: int = 0
    avg_score: float = 0.0
    authors: int = 0
    days: int = 0
    daily: list[tuple[str, int, int]] = field(default_factory=list)
    buckets: list[tuple[str, int]] = field(default_factory=list)
    top_authors: list[tuple[str, int]] = field(default_factory=list)


def _build_analytics(result: PipelineResult) -> _Analytics:
    stories = result.all_stories
    analytics = _Analytics(stories=stories)
    if not stories:
        return analytics

    scores = [story.score or 0 for story in stories]
    analytics.points = sum(scores)
    analytics.avg_score = sum(scores) / len(scores)
    analytics.comments = sum(len(story.comments) for story in stories)
    analytics.authors = len({story.author for story in stories})

    dates = [story.created_at for story in stories if story.created_at]
    if dates:
        total_by_day = Counter(value.date() for value in dates)
        clustered_by_day = Counter(
            story.created_at.date()
            for cluster in result.clusters
            for story in cluster.stories
            if story.created_at
        )
        days = sorted(total_by_day)
        analytics.daily = [
            (
                day.strftime("%m-%d"),
                clustered_by_day.get(day, 0),
                total_by_day[day] - clustered_by_day.get(day, 0),
            )
            for day in days
        ]
        analytics.days = (days[-1] - days[0]).days + 1

    analytics.buckets = _score_buckets(scores)
    analytics.top_authors = Counter(story.author for story in stories).most_common(10)
    return analytics


def _score_buckets(scores: Sequence[int]) -> list[tuple[str, int]]:
    counts = [0] * len(_SCORE_EDGES)
    for score in scores:
        for index in range(len(_SCORE_EDGES) - 1, -1, -1):
            if score >= _SCORE_EDGES[index]:
                counts[index] += 1
                break
    labels = [
        *(
            f"{_SCORE_EDGES[index]}-{_SCORE_EDGES[index + 1] - 1}"
            for index in range(len(_SCORE_EDGES) - 1)
        ),
        f"{_SCORE_EDGES[-1]}+",
    ]
    return list(zip(labels, counts))


def _fmt_value(value: float) -> str:
    if value >= 10000:
        return f"{value / 1000:.0f}k"
    if value >= 1000:
        return f"{value / 1000:.1f}k".replace(".0k", "k")
    return f"{value:g}"


def _hex_rgba(color: str, alpha: float) -> str:
    value = color.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _svg_text(
    x: float,
    y: float,
    content: str,
    *,
    cls: str = "tick",
    anchor: str = "middle",
    rotate: float = 0.0,
) -> str:
    transform = f' transform="rotate({rotate:.0f} {x:.1f} {y:.1f})"' if rotate else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" '
        f'text-anchor="{anchor}"{transform}>{escape(content)}</text>'
    )


def _empty_svg(width: int, height: int) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img">'
        f"{_svg_text(width / 2, height / 2, 'no data')}</svg>"
    )


def _bar_chart_svg(
    bars: Sequence[tuple[str, Sequence[int]]],
    colors: Sequence[str],
    width: int = 720,
    height: int = 300,
) -> str:
    if not bars:
        return _empty_svg(width, height)

    n = len(bars)
    totals = [sum(values) for _, values in bars]
    max_value = max(totals) or 1
    left, right, top, bottom = 44, 10, 18, 34
    plot_w = width - left - right
    plot_h = height - top - bottom
    step = plot_w / n
    bar_w = min(step * 0.66, 52)
    label_every = max(1, math.ceil(n / 14))
    rotate = -40 if n > 8 else 0

    parts = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1 - fraction)
        value = int(max_value * fraction)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" '
            f'y2="{y:.1f}" class="grid"/>'
        )
        parts.append(_svg_text(6, y - 4, _fmt_value(value), anchor="start"))
    parts.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" '
        f'y2="{top + plot_h}" class="axis"/>'
    )

    for index, (label, values) in enumerate(bars):
        x = left + index * step + (step - bar_w) / 2
        y = top + plot_h
        for value, color in zip(values, colors):
            if value <= 0:
                continue
            segment_h = plot_h * value / max_value
            y -= segment_h
            parts.append(
                f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{segment_h:.1f}" rx="2" fill="{color}">'
                f"<title>{escape(f'{label}: {value}')}</title></rect>"
            )
        if totals[index] > 0 and n <= 16:
            parts.append(_svg_text(x + bar_w / 2, y - 6, str(totals[index]), cls="val"))
        if index % label_every == 0:
            parts.append(
                _svg_text(
                    left + index * step + step / 2,
                    height - 10,
                    label,
                    anchor="end" if rotate else "middle",
                    rotate=rotate,
                )
            )
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(parts)}</svg>'


def _hbar_chart_svg(
    rows: Sequence[tuple[str, float, str]],
    width: int = 720,
    label_width: int = 185,
    bar_h: int = 17,
    gap: int = 8,
) -> str:
    if not rows:
        return _empty_svg(width, 60)

    height = 12 + len(rows) * (bar_h + gap)
    max_value = max(value for _, value, _ in rows) or 1
    plot_w = width - label_width - 56

    parts = []
    for index, (label, value, color) in enumerate(rows):
        y = 12 + index * (bar_h + gap)
        bar_w = max(2.0, plot_w * value / max_value)
        parts.append(
            f'<rect class="bar" x="{label_width}" y="{y}" width="{bar_w:.1f}" '
            f'height="{bar_h}" rx="4" fill="{color}">'
            f"<title>{escape(f'{label}: {value:g}')}</title></rect>"
        )
        parts.append(
            _svg_text(label_width - 8, y + bar_h - 5, truncate(label, 26), anchor="end")
        )
        parts.append(
            _svg_text(
                label_width + bar_w + 8,
                y + bar_h - 5,
                _fmt_value(value),
                anchor="start",
                cls="val",
            )
        )
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(parts)}</svg>'


def _scatter_svg(
    stories: Sequence[Story],
    colors: dict[int, str],
    width: int = 960,
    height: int = 560,
) -> str:
    coords: list[tuple[float, float, Story]] = []
    for story in stories:
        if story.pos_x is not None and story.pos_y is not None:
            coords.append((story.pos_x, story.pos_y, story))
    if len(coords) < 2:
        return _empty_svg(width, height)

    pad = 30
    xs = [x for x, _, _ in coords]
    ys = [y for _, y, _ in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = (max_x - min_x) or 1.0
    span_y = (max_y - min_y) or 1.0
    max_score = max((story.score or 0 for _, _, story in coords), default=0) or 1

    parts = []
    for pos_x, pos_y, story in coords:
        x = pad + (pos_x - min_x) / span_x * (width - pad * 2)
        y = pad + (pos_y - min_y) / span_y * (height - pad * 2)
        radius = 3 + 8.5 * math.sqrt((story.score or 0) / max_score)
        color = colors.get(story.cluster_label, _NOISE_COLOR)
        title = escape(f"{story.title} · {story.score or 0} pts")
        parts.append(
            f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
            f'fill="{color}" fill-opacity="0.75"><title>{title}</title></circle>'
        )
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(parts)}</svg>'


def _legend_html(items: Sequence[tuple[str, str]]) -> str:
    if not items:
        return ""
    chips = "".join(
        f'<li><span class="chip" style="background:{color}"></span>{escape(label)}</li>'
        for label, color in items
    )
    return f'<ul class="legend">{chips}</ul>'


class HTMLReportBuilder:
    def __init__(
        self,
        output_path: str | Path | None = None,
        comments_per_story: int | None = None,
        comment_chars: int | None = None,
        charts: bool | None = None,
    ) -> None:
        self._output = Path(output_path or settings.output_html)
        self._comments_per_story = (
            settings.report_comments_per_story
            if comments_per_story is None
            else comments_per_story
        )
        self._comment_chars = (
            settings.report_comment_chars if comment_chars is None else comment_chars
        )
        self._charts = settings.report_charts if charts is None else charts

    def build(self, result: PipelineResult) -> Path:
        colors = {
            cluster.label: _PALETTE[index % len(_PALETTE)]
            for index, cluster in enumerate(result.clusters)
        }
        analytics = _build_analytics(result)
        charts_html = self._render_charts(result, analytics, colors)
        body = "\n".join(
            [
                self._render_header(result),
                self._render_kpis(result, analytics),
                charts_html,
                self._render_toc(result, colors),
                *(
                    self._render_cluster(cluster, index)
                    for index, cluster in enumerate(result.clusters)
                ),
                self._render_outliers(result.outliers),
                self._render_footer(),
            ]
        )
        if charts_html:
            body = "\n".join(
                [body, self._render_viewer(), f"<script>{_VIEWER_JS}</script>"]
            )
        document = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            "<title>Hacker News Clusters</title>\n"
            f"<style>{_CSS}</style>\n"
            "</head>\n"
            f"<body>\n{body}\n</body>\n</html>\n"
        )
        self._output.write_text(document, encoding="utf-8")
        logger.info(f"HTML report saved: {self._output.resolve()}")
        return self._output

    def _render_header(self, result: PipelineResult) -> str:
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        stats = (
            f"{result.total_stories} stories · {len(result.clusters)} clusters · "
            f"{len(result.outliers)} unclassified · {result.elapsed_seconds:.1f}s"
        )
        return (
            '<header class="page">'
            '<h1>Hacker News <span class="accent">Clusters</span></h1>'
            f'<p class="stats">{escape(stats)} · generated {escape(generated)}</p>'
            "</header>"
        )

    def _render_kpis(self, result: PipelineResult, analytics: _Analytics) -> str:
        total = len(analytics.stories) or 1
        noise_pct = 100 * len(result.outliers) / total
        cards = (
            (str(len(analytics.stories)), "stories"),
            (str(len(result.clusters)), "clusters"),
            (f"{len(result.outliers)} ({noise_pct:.0f}%)", "unclassified"),
            (_fmt_value(analytics.comments), "comments"),
            (_fmt_value(analytics.points), "points"),
            (f"{analytics.avg_score:.0f}", "avg score"),
            (str(analytics.authors), "authors"),
            (f"{analytics.days}d", "timespan"),
        )
        items = "".join(
            f'<div class="kpi"><span class="kpi-value">{escape(value)}</span>'
            f'<span class="kpi-label">{escape(label)}</span></div>'
            for value, label in cards
        )
        return f'<div class="kpis">{items}</div>'

    def _render_charts(
        self,
        result: PipelineResult,
        analytics: _Analytics,
        colors: dict[int, str],
    ) -> str:
        if not self._charts or not analytics.stories:
            return ""

        daily_bars = [
            (label, (clustered, unclassified))
            for label, clustered, unclassified in analytics.daily
        ]
        figures = [
            self._figure(
                _bar_chart_svg(daily_bars, ("var(--accent)", _NOISE_COLOR)),
                "Stories per day",
                legend=_legend_html(
                    [("clustered", "var(--accent)"), ("unclassified", _NOISE_COLOR)]
                ),
                wide=len(analytics.daily) > 12,
            ),
            self._figure(
                _bar_chart_svg(
                    [(label, (count,)) for label, count in analytics.buckets],
                    ("var(--accent)",),
                ),
                "Score distribution",
            ),
            self._figure(
                _hbar_chart_svg(
                    [
                        (author, count, "var(--accent)")
                        for author, count in analytics.top_authors
                    ]
                ),
                "Top authors",
            ),
            self._figure(
                _hbar_chart_svg(
                    [
                        (story.title, story.score or 0, "var(--accent)")
                        for story in sorted(
                            analytics.stories,
                            key=lambda s: s.score or 0,
                            reverse=True,
                        )[:_MAX_TOP_STORIES]
                    ]
                ),
                "Top stories by score",
            ),
        ]

        sentiment_counts = Counter(
            cluster.summary.sentiment
            for cluster in result.clusters
            if cluster.summary and cluster.summary.sentiment
        )
        if sentiment_counts:
            sentiment_bars = [
                (
                    name,
                    tuple(
                        sentiment_counts[name] if name == series else 0
                        for series in _SENTIMENTS
                    ),
                )
                for name in _SENTIMENTS
            ]
            figures.append(
                self._figure(
                    _bar_chart_svg(sentiment_bars, _SENTIMENT_COLORS),
                    "Cluster sentiment",
                )
            )

        if result.clusters:
            shown = result.clusters[:_MAX_CHART_CLUSTERS]
            note = (
                f" · top {len(shown)} of {len(result.clusters)}"
                if len(shown) < len(result.clusters)
                else ""
            )
            labels = [truncate(cluster.display_title, 26) for cluster in shown]
            for caption, key in (
                ("Cluster sizes", "size"),
                ("Cluster points", "total_score"),
                ("Cluster comments", "total_comments"),
            ):
                figures.append(
                    self._figure(
                        _hbar_chart_svg(
                            [
                                (label, getattr(cluster, key), colors[cluster.label])
                                for label, cluster in zip(labels, shown)
                            ]
                        ),
                        f"{caption}{note}",
                    )
                )
            figures.append(
                self._figure(
                    _hbar_chart_svg(
                        [
                            (label, round(cluster.velocity, 2), colors[cluster.label])
                            for label, cluster in zip(labels, shown)
                        ]
                    ),
                    f"Cluster momentum (median points/hour){note}",
                )
            )

        if any(story.pos_x is not None for story in analytics.stories):
            figures.append(
                self._figure(
                    _scatter_svg(analytics.stories, colors),
                    "Cluster map (UMAP projection, circle size = score)",
                    legend=self._render_legend(result, colors),
                    wide=True,
                )
            )

        return (
            '<h2 class="section">Analytics</h2>'
            '<p class="hint">Click a chart to enlarge · ← → switch · '
            "F fullscreen · ⤓ download SVG/PNG</p>"
            f'<div class="chart-grid">{"".join(figures)}</div>'
        )

    @staticmethod
    def _figure(svg: str, caption: str, legend: str = "", wide: bool = False) -> str:
        cls = "chart wide" if wide else "chart"
        return (
            f'<figure class="{cls}"><figcaption>{escape(caption)}</figcaption>'
            f'<div class="chart-body">{svg}{legend}</div></figure>'
        )

    def _render_legend(self, result: PipelineResult, colors: dict[int, str]) -> str:
        items = [
            (truncate(cluster.display_title, 36), colors[cluster.label])
            for cluster in result.clusters[:_MAX_LEGEND_ITEMS]
        ]
        hidden = len(result.clusters) - _MAX_LEGEND_ITEMS
        if hidden > 0:
            items.append((f"+{hidden} more", _NOISE_COLOR))
        items.append(("unclassified", _NOISE_COLOR))
        return _legend_html(items)

    @staticmethod
    def _render_viewer() -> str:
        return (
            '<dialog id="chart-viewer">'
            '<div class="viewer-head">'
            '<h3 class="viewer-title"></h3>'
            '<button type="button" data-act="prev" title="Previous (←)">‹</button>'
            '<button type="button" data-act="next" title="Next (→)">›</button>'
            '<button type="button" data-act="fs" title="Fullscreen (F)">⤢</button>'
            '<button type="button" data-act="svg" title="Download SVG">⤓ SVG</button>'
            '<button type="button" data-act="png" title="Download PNG">⤓ PNG</button>'
            '<button type="button" data-act="close" title="Close (Esc)">✕</button>'
            "</div>"
            '<div class="viewer-body"></div>'
            "</dialog>"
        )

    def _render_toc(self, result: PipelineResult, colors: dict[int, str]) -> str:
        if not result.clusters:
            return ""
        links = "".join(
            f'<a href="#cluster-{cluster.label}">'
            f'<span class="chip" style="background:{colors[cluster.label]}"></span>'
            f"{escape(truncate(cluster.display_title, 40))}</a>"
            for cluster in result.clusters[:_MAX_TOC_ITEMS]
        )
        return f'<nav class="toc">{links}</nav>'

    def _render_cluster(self, cluster: StoryCluster, index: int) -> str:
        color = _PALETTE[index % len(_PALETTE)]
        summary = cluster.summary
        badges = ""
        if summary:
            for cls, value in (
                ("sentiment", summary.sentiment),
                ("momentum", summary.momentum),
            ):
                if value:
                    badges += (
                        f'<span class="{cls} s-{escape(value)}">{escape(value)}</span>'
                    )
        meta_primary = (
            f"{cluster.size} stories · {cluster.total_comments} comments · "
            f"{_fmt_value(cluster.total_score)} points · "
            f"avg {cluster.avg_score:.0f} · median {cluster.median_score:.0f}"
        )
        meta_secondary = (
            f"velocity {cluster.velocity:.2f} pts/h · "
            f"age ~{cluster.age_hours:.0f}h · "
            f"{cluster.fresh_share:.0%} within 48h · "
            f"{cluster.unique_authors} authors"
        )
        if cluster.top_domains:
            domains = ", ".join(map(escape, cluster.top_domains))
            meta_secondary += f" · {domains}"

        terms_html = ""
        if cluster.top_terms:
            chips = "".join(
                f'<li class="term">{escape(term)}</li>' for term in cluster.top_terms
            )
            terms_html = f'<ul class="terms">{chips}</ul>'

        top_stories = sorted(cluster.stories, key=lambda s: s.score or 0, reverse=True)
        stories_html = "\n".join(self._render_story(s) for s in top_stories)
        description = (
            f'<p class="description">{escape(summary.description)}</p>'
            if summary
            else ""
        )
        return (
            f'<section class="cluster" id="cluster-{cluster.label}" '
            f'style="--c:{color};--c-bg:{_hex_rgba(color, 0.1)}">'
            f'<header><span class="rank">#{index + 1}</span>'
            f"<h2>{escape(cluster.display_title)}</h2>{badges}"
            f'<p class="meta">{escape(meta_primary)}</p>'
            f'<p class="meta">{meta_secondary}</p>'
            f"{terms_html}</header>"
            f"{description}"
            f"{stories_html}"
            "</section>"
        )

    def _render_story(self, story: Story) -> str:
        link = story.url or story.hn_url
        date = story.created_at.strftime("%Y-%m-%d %H:%M") if story.created_at else "—"
        comments_html = "".join(
            self._render_comment(comment)
            for comment in story.comments[: self._comments_per_story]
        )
        details = (
            f"<details><summary>Show comments</summary>{comments_html}</details>"
            if comments_html
            else ""
        )
        meta = f"{story.score or 0} points · by {escape(story.author)} · {escape(date)}"
        return (
            '<article class="story">'
            f'<a class="title" href="{escape(link, quote=True)}" '
            f'target="_blank" rel="noopener">{escape(story.title)}</a> '
            f'<a class="discussion" href="{escape(story.hn_url, quote=True)}" '
            f'target="_blank" rel="noopener">[discussion]</a>'
            f'<p class="meta">{meta}</p>'
            f"{details}"
            "</article>"
        )

    def _render_comment(self, comment: Comment) -> str:
        author = escape(comment.author or "anon")
        text = escape(truncate(strip_html(comment.text), self._comment_chars))
        return f'<div class="comment"><b>{author}</b>: {text}</div>'

    @staticmethod
    def _render_outliers(outliers: list[Story]) -> str:
        if not outliers:
            return ""
        items = "\n".join(
            f'<li><a href="{escape(story.hn_url, quote=True)}" target="_blank" '
            f'rel="noopener">{escape(story.title)}</a> '
            f"({story.score or 0} pts, by {escape(story.author)})</li>"
            for story in outliers
        )
        return (
            '<details class="outliers"><summary>'
            f"Unclassified stories ({len(outliers)})</summary>"
            f"<ul>{items}</ul></details>"
        )

    @staticmethod
    def _render_footer() -> str:
        return '<footer class="page">Generated by hn-sentiment-analysis</footer>'
