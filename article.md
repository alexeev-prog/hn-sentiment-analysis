# Я проанализировал 100 000 постов с Hacker News и вот что хочу сказать
Hacker News — это пульс мирового технологического сообщества. Тысячи разработчиков, основателей стартапов, инженеров из крупных компаний ежедневно делятся здесь инсайтами, проектами, новостями и мемами. Но есть проблема: качественный контент тонет в шуме, или интересные тебе темы тонут в неинтересных тебе темах.

Но из-за большого потока постов качественные статьи по нужной теме теряются среди других, и если нужно найти какие-то системные тренды, увидеть, как меняется отношение к Rust, AI или Open Source, а не просто «почитать что-то интересное» — это сделать сложно. Можно, да, скопировать кучу ссылок и отправить в LLM, попросив отсортировать, но это занимает время и не так удобно, если нужно проанализировать большое количество тем.

И тут мне пришла идея — создать программу на Python, которая будет парсить через официальный API Hacker News, строить эмбеддинги постов с комментариями, кластеризовать посты и при помощи LLM суммаризировать этот кластер по ссылкам.

Сказано — сделано. Я смог спарсить 100 000 постов с Hacker News за последние 693 дня, получив более тысячи кластеров, более 10 миллиона апвоутов, около 27 тысяч авторов и всего 0.1% выбросов. В этой статье мы разберем, какие тренды были активны в мировом IT-сообществе последние два года.

---

Итак, однажды, в процессе скроллинга Hacker News в поиске вдохновения и интересных статей, я задумался: у Hacker News есть целых два API (Firebase и Algolia), так почему бы не придумать какой-нибудь проект с использованием API? Но обычный постинг в условный телеграм-канал или просто приложение показался мне скучным. Немного подумав, я пришел к идее: а почему бы не автоматизировать и не проанализировать посты на Hacker News? Это и идеи для статей, и возможность почитать и проанализировать мнения по разным темам, найти интересующие вещи.

Я начал думать, какие технологии пригодятся. Просто кидать LLM ссылки и просить выделить темы — скучно. Я решил пойти путем другим, через создание ETL (Extract-Transform-Load) пайплайна. Сначала мы парсим сами статьи, и если позволяет rate-limit — то и 3–5 комментариев к ним. После мы формируем из них объекты данных, которые в последующем через трансформер all-MiniLM-L6-v2 преобразуем в эмбеддинги. Эти самые эмбеддинги кластеризируются через HDBSCAN, а затем LLM анализирует и создает заголовок и саммари для каждого из кластеров. И в конце — создаем JSON и HTML-файл, графики различные. Для работы с LLM я использовал [BotHub API](https://bothub.chat/ru/profile/for-developers).

Конечно, в процессе разработки я сталкивался с проблемами, например: оптимизация запросов к API, настройка кластеризации с маленьким выбросом (то есть некластеризированными статьями), работа с LLM через API, конфигурация и сам пайплайн. Но итоговый результат того стоил — за несколько минут мы получаем двухлетний срез всех мнений в IT-сообществе.

Сам исходный код программы, которую я разработал для сбора данных, вы можете [прочитать в моем репозитории](https://github.com/alexeev-prog/hn-sentiment-analysis). Перед тем как начать разбор, хочу сказать, что я также сделал онлайн-карту историй, доступную [по этой ссылке](https://alexeev-prog.github.io/hn-sentiment-analysis/). Для анализа требуется файл кластеризации: вы можете сгенерировать его сами через мой репозиторий или скачать [готовый файл кластеризации hn_clusters.json](https://github.com/alexeev-prog/hn-sentiment-analysis/releases/tag/1.0).

В этой статье я разберу, какую работу я проделал, а также проанализирую графики, кластеры и статистику, и узнаем, что было на уме в IT-сфере последние годы.

# Архитектура решения

Я решил создать ETL-пайплайн для того, чтобы грамотно собрать все данные и кластеризировать всё. Пайплайн состоит из четырёх логических этапов: извлечение, трансформация, кластеризация и суммаризация.

![](https://habrastorage.org/webt/ff/44/3e/ff443eadc4032f86f896203acd129813.png)

Начнем с этапа извлечения данных. Тут довольно интересно: Hacker News имеет два официальных API — [Firebase Hacker News API](https://github.com/HackerNews/API) и [Algolia API](https://hn.algolia.com/api). Firebase дает прямой доступ к сырым данным в реальном времени. Отлично подходит для мониторинга новых постов, но для анализа вглубь, увы, неудобен: нужно проходиться по всем ID, делать сотни запросов к `/item/`, фильтрация и сортировка — на стороне клиента. А вот Algolia удобен и своим поисковым движком, и готовыми индексами. Он позволяет фильтровать, сортировать, ограничивать по дате и скору, и поэтому я выбрал его.

Но у Algolia есть лимиты и ограничения, например не более 1000 постов за раз.

У меня в коде, в части связанной с фильтрацией, есть параметр `search_by_date`. Если он `False`, то мы используем API-эндпоинт `https://hn.algolia.com/api/v1/search`, и он отдает по релевантности. А по умолчанию он равен `True`, и используется уже API-эндпоинт `https://hn.algolia.com/api/v1/search_by_date`. Вместо того чтобы запрашивать страницы подряд, я использую параметр `created_at_i<{timestamp}` и двигаюсь назад во времени. Механика простая: запросили 100 постов с датой меньше текущей, взяли минимальную дату из ответа, подставили ее в следующий запрос. Так можно собрать хоть 100 000 постов — пока не упремся в `max_pages` или не кончатся данные.

Также есть и фильтрация через `QueryParams`, который строит словарь для передачи в API.

```python
class QueryParams(BaseModel):
    query: str | None = None
    tags: list[str] | None = None
    numeric_filters: str | None = None
    filters: str | None = None
    hitsPerPage: int = 100
    page: int = 0

    def build_dict(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for field in self.model_fields:
            value = getattr(self, field)
            if value is None:
                continue
            if isinstance(value, list):
                value = ",".join(value)
            params[_PARAM_ALIASES.get(field, field)] = value
        return params
```

Кроме того, есть `AlgoliaStoryFilter` — это абстрактный класс для удобного создания различных фильтров:

```python
class AlgoliaStoryFilter(ABC):
    @abstractmethod
    def apply(self, stories: list[Story]) -> list[Story]: ...

    def numeric_filters(self) -> list[str]:
        return []

    def query_tags(self) -> list[str]:
        return []


class DateRangeFilter(AlgoliaStoryFilter):
    ...

class MinScoreFilter(AlgoliaStoryFilter):
    ...


class AuthorFilter(AlgoliaStoryFilter):
    ...


class KeywordFilter(AlgoliaStoryFilter):
    ...
```

А также есть сортировка:

```python
class StorySorter:
    @staticmethod
    def by_score(stories: list[Story], reverse: bool = True) -> list[Story]:
        return sorted(stories, key=lambda s: s.score or 0, reverse=reverse)

    @staticmethod
    def by_date(stories: list[Story], reverse: bool = True) -> list[Story]:
        return sorted(
            stories,
            key=lambda s: s.created_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=reverse,
        )

    @staticmethod
    def by_alphabetical(stories: list[Story], reverse: bool = False) -> list[Story]:
        return sorted(stories, key=lambda s: s.title or "", reverse=reverse)
```

После загрузки постов я параллельно подгружаю комментарии — отдельным запросом с `tags=comment` и `filters=story_id={id}`. Максимальное число комментариев на пост регулируется через `max_comments_per_story`, чтобы не перегружать LLM на этапе суммаризации.

Всё это работает уже через наши DTO: `Story` и `Comment`:

```python
class Comment(BaseModel):
    id: int
    text: str
    created_at: str
    author: str | None = None
    story_id: int | None = None

    @property
    def clean_text(self) -> str:
        return strip_html(self.text)

    @property
    def length(self) -> int:
        return len(self.clean_text)


class Story(BaseModel):
    id: int
    title: str
    url: str | None
    created_at: datetime | None
    author: str
    score: int
    tags: list[str] = Field(default_factory=list)
    comments: list[Comment] = Field(default_factory=list)
    embedding: list[float] | None = None
    cluster_label: int | None = None
    pos_x: float | None = None
    pos_y: float | None = None

    @property
    def hn_url(self) -> str:
        return f"https://news.ycombinator.com/item?id={self.id}"

    @property
    def embedding_text(self) -> str:
        parts = [self.title] * settings.embedding_title_repeats
        for comment in self.comments[: settings.embedding_max_comments]:
            snippet = comment.clean_text[: settings.embedding_comment_chars]
            if snippet:
                parts.append(snippet)
        return "\n".join(parts)[:_MAX_EMBEDDING_TEXT_CHARS]
```

Комменты предварительно также очищаются от HTML-символов, чтобы не было мусора во время превращения в вектор.

Следующий этап — трансформация, то есть превращение данных в векторы, эмбеддинги.

Текст эмбеддинга формируется просто: заголовок повторяется три раза (самый сильный сигнал, чтобы комментарии не могли увести куда-то в сторону). Далее добавляем комментарии с обрезкой до 350 символов.

Как трансформер я использую `all-MiniLM-L6-v2`. Модель быстрая и качественная, выдает нормализованные эмбеддинги (`normalize_embeddings=True`), что критично для кластеризации через косинусное расстояние. Да, можно воспользоваться более качественными решениями или API, но они медленные, а в случае огромной массы историй скорость важна.

Следующий этап — кластеризация. Тут выбор был между HDBSCAN и K-means. Я выбрал HDBSCAN, так как у KMeans нужно было знать итоговое количество кластеров, а у нас неизвестно. А также он умеет маркировать шум (noise) — посты, которые не вписались ни в один кластер.

Перед HDBSCAN я применяю UMAP с параметрами:

> UMAP (Uniform Manifold Approximation and Projection) — это алгоритм снижения размерности и визуализации многомерных данных.

```python
umap.UMAP(
    n_components=12,       # редуцируем 384 → 12
    n_neighbors=10,        # локальный контекст
    min_dist=0.1,          # плотность точек
    metric="cosine",       # расстояние для эмбеддингов
)
```

Редукция до 12 измерений является компромиссом: сохраняет структуру данных, но ускоряет кластеризацию. В коде также сохраняются первые две компоненты `(points[:, :2])` для визуализации на карте.

А теперь к обработке выбросов. HDBSCAN обычно оставляет 10–30% точек как шум. Моя задача — минимизировать потери и стараться прийти к тому, чтобы все посты были распределены по кластерам (и чтобы кластеров было достаточно много). В коде реализован алгоритм присвоения выбросов:

```python
# Нормализуем эмбеддинги (L2-норма)
data = embeddings / np.linalg.norm(data, axis=1, keepdims=True)

# Считаем центроиды кластеров (средний вектор)
centroids = [data[labels == cluster_id].mean(axis=0) for cluster_id in cluster_ids]

# Косинусное сходство каждой шумовой точки с центроидами
similarities = data[noise_positions] @ centroids.T

# Присваиваем, если сходство >= 0.45 (cluster_outlier_threshold)
accepted = similarities.max(axis=1) >= 0.45
labels[noise_positions[accepted]] = cluster_ids[best[accepted]]
```

Порог 0.45 отсекает точки, которые слишком далеки от любого кластера. В результате из 100 000 постов только 0.11% остались некластеризованными. Это говорит о том, что UMAP + HDBSCAN с правильными параметрами (плюс релевантная выборка) отлично разделяет темы.

## Суммаризация

Кластеры, безусловно, это очень хорошо. Но мне показалось, просто смотреть сырые данные может быть трудозатратным, а то и вовсе скучным. Поэтому я решил реализовать суммаризацию через API к LLM, через openai-совместимый шлюз.

У нашего сервиса [BotHub](реферальной ссылке](https://bothub.ru/?invitedBy=zx7xIePYIOcWHhTfgg6-O) есть несколько преимуществ: можно оплачивать тарифы с российских банковских карт и иметь доступ практически ко всем нейросетям, включая новейшие, по низким ценам (иногда даже ниже, чем OpenRouter). Пайплайн и чат-бот в онлайн-карте работают через BotHub, и вы можете получить 300 тысяч бесплатных CAPS, перейдя по [реферальной ссылке](https://bothub.ru/?invitedBy=zx7xIePYIOcWHhTfgg6-O), если захотите сами запустить пайплайн.

 > Используя модель gpt-5.6 luna, я потратил всего ~0.09$ (1 069 550 токенов) за один полный прогон пайплайна.

![](https://habrastorage.org/webt/4b/b7/91/4bb79152e7b206f75484a431fca5cacc.png)

После регистрации для получения API-ключа достаточно приобрести CAPS (внутренняя валюта для токенов) и зайти в [раздел для разработчиков](https://bothub.chat/ru/profile/for-developers). Документацию по ней можно найти [здесь](https://bothub.chat/api/documentation/ru). Также API поддерживает [Batches API](https://bothub.chat/api/documentation/ru/generation/batches).

![](https://habrastorage.org/webt/3f/13/20/3f13207c6f3795510d363ccd2cde081e.png)

Получив токен, вы можете обращаться через библиотеку openai, поставив как url `https://openai.bothub.chat/v1` и вставив свой токен:

```python
self._client = AsyncOpenAI(
    api_key=key,
    base_url=base_url or settings.ai_base_url,
    timeout=settings.ai_request_timeout,
)
```

Я взял за основу модель [gpt-5.6 luna](https://bothub.chat/gpt-5.6-luna). Она дешевая и быстрая, а для наших требований больше и не надо.

На вход модели подается сжатое представление кластера:

```python
for story in top_stories[:ai_max_stories_in_prompt]:  # 30 постов по умолчанию
    lines.append(f"- ({story.score} points) {story.title}")
    for comment in story.comments[:ai_max_comments_in_prompt]:  # 3 комментария
        snippet = truncate(comment.clean_text, ai_prompt_comment_chars)  # 338 символов
```

Чтобы парсить ответы машинно, в system-промпте жестко задан формат:

```python
system = (
    "You are an expert analyst of Hacker News discussions. You receive "
    f"{len(batch)} cluster(s) of related posts with comment snippets. "
    "Answer ONLY with a valid minified JSON array containing exactly one "
    f"object per cluster, in input order, in the format {_JSON_SPEC}. "
    f"Write all fields in {settings.ai_summary_language}. For each "
    "cluster: 'index' repeats its cluster number; 'title' must name the "
    "concrete shared topic (a product, technology, company or event) — "
    "never a vague heading like 'Tech Discussions'; 'description' must "
    "state what unites the posts, the community's dominant opinion, and "
    "any notable disagreement or concern; 'sentiment' is the overall "
    "tone of the discussion: 'positive', 'negative', 'mixed' when "
    "opinions clearly split, or 'neutral' for factual discussions. "
    "No markdown, no extra keys, no text outside the JSON array."
)
return [
    {"role": "system", "content": system},
    {"role": "user", "content": "\n\n".join(sections)},
]
```

Тональность (sentiment) вычисляется на основе дискуссии внутри кластера, она может быть `positive` (хвалят и восторгаются), `negative` (критикуют), `mixed` (мнения разделены), `neutral` (нейтрально).

Через пайплайн в итоге формируется HTML-отчёт и JSON-файл с данными кластеризации (его вы как раз и можете загрузить на [моём сайте](https://alexeev-prog.github.io/hn-sentiment-analysis/)).

```json
{
  "metadata": {
    "total_stories": 100000,
    "elapsed_seconds": 3249.432794319,
    "clusters_count": 1272,
    "outliers_count": 103,
    "generated_at": "2026-09-05T23:02:31.692155"
  },
  "clusters": [
    {
      "label": 878,
      "size": 575,
      "total_score": 87277,
      "total_comments": 10,
      "avg_score": 151.78608695652173,
      "display_title": "Large Language Model Debate",
      "summary": {
        "title": "Large Language Model Debate",
        "description": "The posts examine how LLMs work, their coding capabilities, practical workflows, risks, and effects on software careers. The community is deeply engaged but divided, with enthusiasm about learning and productivity balanced by concerns about hallucinations, poisoning, limited reasoning, and displacement.",
        "sentiment": "mixed",
        "model": "gpt-5.6-luna"
      },
      "stories": [...]
    }
    ...
  ]
}
```

# Результаты

Пора перейти к самому интересному — какие были получены результаты? Я проанализировал ровно 100 000 последних постов на Hacker News и получил:

- 1272 кластера
- 103 неклассифицированных
- 3400 комментариев
- 10 412 апвоутов
- в среднем 104 апвоута на пост
- 27 223 автора
- за 694 дня
- 1 069 550 токенов потрачено

1 069 550 в моем случае стоило всего около 0.09 $ (если быть точными, то 0.09209)! В пересчете на кластер, это всего лишь 0.000072398 $ на кластер!

Давайте разберем графики. Первый — это количество постов в день. Если внимательно приглядеться, можно заметить постепенный рост популярности Hacker News через большее количество постов.

![Количество постов в день](https://habrastorage.org/webt/da/79/ed/da79ed44da3433f4387e72ae7da336de.png)

Чтобы отсортировать малопопулярные статьи, я выставил фильтрацию по `score > 10`, так что можно увидеть распределение. Естественно, с 10 до 24 апвоутов — самый большой столбец, с 25 до 499 — примерно от 20 до 10 тысяч постов, а 500+ набрали всего лишь около 3 тысяч постов.

![Распределение апвоутов](https://habrastorage.org/webt/97/cc/81/97cc81bbae4befb58b1a68c182c768cd.png)

Также видно на графике распределения кластеров по настроению, что «позитивных» постов чуть больше, чем «негативных», но смешанных намного больше, а нейтральных — меньшинство.

![Распределение кластеров по настроению](https://habrastorage.org/webt/0a/2c/7a/0a2c7a8863274fca4a098a6c092ff92e.png)

А ниже вы можете увидеть саму карту кластеров:

![Карта кластеров](https://habrastorage.org/webt/0d/0d/69/0d0d69f0d9c7c832ee21d069a496ad1e.png)

И как видно по таблице ниже, почти все топ-посты так или иначе связаны с ИИ и LLM.

![Посты](https://habrastorage.org/webt/3f/11/33/3f11336a07e08ef9f8fafebd21e63f1f.png)

Крупнейшие кластеры видны на графике ниже. Опять же, половина связана с ИИ (6 из крупнейших кластеров). А также виден большой интерес к опенсорсу!

![Крупнейшие кластеры](https://habrastorage.org/webt/8c/5f/79/8c5f795aae77844a6a6e8b1b38539e73.png)

А если посмотреть по скору, то видно, что на первом месте опять LLM и опять 6 кластеров связаны с ИИ. Но остальные уже более технические, связанные с проектами, чаще всего опенсорса.

![Самые горячие кластеры](https://habrastorage.org/webt/87/95/48/8795481ed59e303465f4708c849bf266.png)

Ниже график распределения размеров кластеров. Ось Y — количество кластеров, ось X — количество постов.

![Распределение размеров кластеров](https://habrastorage.org/webt/84/c2/e2/84c2e2d4b15b6f02c1199e35c6d5f6f1.png)

Ниже — топ авторов по количеству постов.

![Топ авторов по количеству историй](https://habrastorage.org/webt/93/2b/f1/932bf11b6885e474d6fa9093954ba282.png)

А ниже — топ источников, опубликованных на Hacker News. Здесь расположился на первом месте с большим отрывом GitHub.

![](https://habrastorage.org/webt/aa/6c/c7/aa6cc78f233c1d86393a4e753f3d25ea.png)

Если рассмотреть дни недели, видно, что в основном статьи на Hacker News выходят в будние дни.

![Распределение постов по неделям](https://habrastorage.org/webt/05/7a/de/057adec0337074535385e479b21e2849.png)

А если посмотреть распределение по часам, видно, что большинство постов публикуются с 12 до 24 часов, и основной поток — днём.

![Распределение по часам](https://habrastorage.org/webt/46/1f/17/461f171ce98e130fcce638aee1513234.png)

А вот ниже можно рассмотреть тональность по месяцам (красная — негативная, зелёная — позитивная, жёлтая — смешанная, серая — нейтральная). Интересно, что позитивная линия и негативная линия идут рука об руку, кроме периода с 02.25 до 04.25. А нейтральная и смешанная всегда примерно на одном уровне.

![Тональность по месяцам](https://habrastorage.org/webt/f2/92/aa/f292aa37ade59a01006855ef00632a76.png)

А теперь давайте разберем конкретные тренды и кластеры!

# Тренды мирового ИТ за 2 года

В этом блоке я разберу самые интересные крупнейшие кластеры. Вы можете сами изучить все остальные кластеры на [моём сайте](https://alexeev-prog.github.io/hn-sentiment-analysis/).

## AI Fatigue And Cognitive Costs

AI — безусловный лидер обсуждений. Но тут мнения совершенно полярные. Одни критикуют, и нарастающий кластер «AI Fatigue» показывает, что люди устали от хайпа, начинают критиковать качество, стоимость и этику ИИ.

![Практически весь кластер AI Fatigue And Cognitive Costs негативный!](https://habrastorage.org/webt/27/f7/60/27f76063c75b51d868b00533f56dad76.png)

![AI Fatigue And Cognitive Costs становится популярнее совместно с ростом популярности ИИ](https://habrastorage.org/webt/59/75/41/5975415f2f95004bb12abfa624e9916a.png)

Как мы видим на графиках, это очень яркий кластер, который посвящён ругани на ИИ. Основные тезисы этого кластера заключаются в том, что качество AI-сгенерированной работы снижается, появляется необходимость проверять результат. Появляются когнитивные издержки в виде синдрома самозванца, ослабления понимания и компетенций. А также интересно, что акцентируется внимание на ИИ-агентах и желании популяризировать локальные модели. Кроме технических, упоминается и просто усталость и перегрузка от AI-тематики. Встречаются и контраргументы: аккуратное применение AI действительно может повышать продуктивность.

![Статистика кластера AI Fatigue And Cognitive Costs](https://habrastorage.org/webt/94/64/84/94648452b1cf0f315a8b6bc745688cb3.png)

## AI Progress and Social Impact

Кластер, похожий на предыдущий, но более сфокусирован на влиянии на социум и развитие. Обсуждаются темпы и пределы развития искусственного интеллекта, а также слежка, нарушения на рабочем месте, атрофия навыков, риски для безопасности, зависимость и негативная реакция общественности. Преобладающий тон — скептический и тревожный, хотя некоторые авторы утверждают, что прогресс реален, формальная проверка и другие области принесут пользу, а оппозиция не должна превращаться в огульную шумиху против искусственного интеллекта. Это показывает самая горячая статья — [Don't fall into the anti-AI hype](https://antirez.com/news/158).

![](https://habrastorage.org/webt/68/8c/e6/688ce673ecf21534a0d4a8cce9778ee4.png)

Кстати, если сравнить с предыдущим кластером, можно увидеть следующее:

![](https://habrastorage.org/webt/24/df/31/24df312591ccb32605e6b601851d318c.png)

## Large Language Model Debate

![](https://habrastorage.org/webt/5b/2b/02/5b2b021f77bd4259eca869c53f5d3971.png)

Крупнейший кластер. Если верить ИИ, делающему суммаризацию этого кластера:

> The posts examine how LLMs work, their coding capabilities, practical workflows, risks, and effects on software careers. The community is deeply engaged but divided, with enthusiasm about learning and productivity balanced by concerns about hallucinations, poisoning, limited reasoning, and displacement.

Тональность здесь смешанная, с высокой вовлечённостью и заметным расколом мнений. Главные темы — это принципы работы LLM, программирование с их помощью и их ограничения в виде галлюцинаций (кстати, вы можете почитать в нашем блоге две статьи на тему галлюцинаций нейросети: [часть 1](https://habr.com/ru/companies/bothub/articles/1076364/) и [часть 2](https://habr.com/ru/companies/bothub/articles/1076472/)). Также в этом кластере есть обсуждения на тему обучения, открытой инфраструктуры и применения LLM в проектах.

![](https://habrastorage.org/webt/91/db/bf/91dbbf2f8d52edb121c4870db5494d3d.png)

Также тут топ-источник — не GitHub, а arXiv.

## Show HN Indie Projects

Есть и позитивный кластер в виде постов под тематикой Show HN.

![](https://habrastorage.org/webt/85/6b/1d/856b1d9a81999cb53d19ade1c5719cca.png)

Сообщество в целом поддерживает и обсуждает независимые проекты. Главные темы тут — инди-игры, творческие эксперименты, утилиты, образовательные инструменты и персональные проекты. Дискуссия [«Is Show HN dead? No, but it's drowning»](https://www.arthurcnops.blog/death-of-show-hn/), несмотря на критический заголовок, показывает, что есть и критическое видение.

## Open Source Licensing And Funding

Так как Open Source — неотъемлемая часть IT-сообщества, эта тема есть и в кластерах. Тут рассказывается о проектах с открытым исходным кодом, лицензиях, бесплатных инструментах для разработчиков, поддержке мейнтейнеров. Настроения в целом позитивны, но опасения по поводу прав на повторную выдачу лицензий, моделей, основанных на источниках, корпоративного поведения, финансирования и устойчивости создают некоторую напряжённость.

![](https://habrastorage.org/webt/6d/0f/41/6d0f414a12b2a02f73f52db9fdc950b9.png)

Ну и стоит помнить, что Hacker News организован YC — успешным стартап-акселератором. И поэтому лидер выборки — [история о релицензировании проекта компанией из YC](https://twitter.com/soham_btw/status/1940952786491027886).

Но есть разрыв между средним и топовыми историями: средний скор — 126, тогда как максимум — 947.

## Python Language Evolution

Этот кластер даёт понять, что Python активно живёт и развивается и не планирует сдавать позиции.

![](https://habrastorage.org/webt/91/a2/39/91a2393861fe5f23e7211487ac05ca42.png)

Посты здесь сфокусированы на Python 3.14 и 3.15. Поднимаются темы субинтерпретаторов, тайп-чекеров, пакетников, lazy-импортов, freethreading. Энтузиазм за модернизацию идёт рука об руку со спорами о типизации в Python и переусложнении. Также есть дискуссии об экосистеме Python и проектах на нём.

Самая горячая история здесь — это [Ty: A fast Python type checker and language server](https://github.com/astral-sh/ty), новый проект от astral-sh, создателей uv.

Среди источников можно достаточно много найти ссылок на дискуссии, форумы, документацию Python и связанными с ним проектами.

![](https://habrastorage.org/webt/c7/44/86/c74486602ef8e65e08e88e8396a06f77.png)

## Experimental Programming Languages

Неожиданно, но разработка языков программирования интересна многим разработчикам. Да и в последнее время много внимания нетрадиционным языкам, индустрия постепенно меняется.

![](https://habrastorage.org/webt/f5/d1/12/f5d1121e09ef975a77427d7f5235e6b4.png)

В статьях рассматриваются новые и необычные языки программирования, принципы проектирования языков, системы типов, компиляторы, визуальное программирование и исторические влияния. Верхушка выборки сильно сосредоточена вокруг языкового дизайна и историко-концептуальных материалов, а не только вокруг анонсов новых языков.

## Rust Systems Programming

Также показывает популярность и язык программирования Rust. В этом кластере рассказывается о Rust, инструментах, компиляторах, базах данных, улучшениях безопасности и внедрении в системное программное обеспечение. Преобладает энтузиазм по поводу безопасности и производительности Rust, но остаются разногласия по поводу сложности асинхронности, рисков для экосистемы, совместимости и того, можно ли широко заменить C или C++.

![](https://habrastorage.org/webt/7a/7a/65/7a7a651bc8b8dff5dc2d19a488bcd3bd.png)

Самый горячий пост тут — [Postgres rewritten in Rust, now passing 100% of the Postgres regression tests](https://github.com/malisper/pgrust).

Сам кластер довольно растущий по трендам, хоть и фанатизм пропадать стал. Высокий интерес вызывают критические материалы: [«Async Rust never left the MVP state»](https://tweedegolf.nl/en/blog/237/async-rust-never-left-the-mvp-state), а также [возврат с Rust на C++](https://old.reddit.com/r/rust/comments/1h15md8/goodbye_rust_i_wish_you_success_but_im_back_to_c/).

![](https://habrastorage.org/webt/98/a7/9d/98a79db865702d5f2556a0d6b0a634e2.png)

## Go Programming Language

Практически сразу за Rust идёт кластер про Go, где рассказывается о его релизах, ошибках компилятора, инструментах, обобщениях, производительности, безопасности, разработке с использованием искусственного интеллекта и практическом использовании экосистемы. Сообщество ценит простоту и продуктивность Go, но периодически критикует языковые ограничения и компромиссы в дизайне.

![](https://habrastorage.org/webt/45/a4/33/45a433e692a8e9030371b32a7956a88d.png)

## Lisp Programming Ecosystem

Совершенно неожиданно, но относительно большой кластер связан с Lisp, да ещё и кластер позитивный (а Rust, Go и Python были смешанными). В публикациях отмечаются выразительный дизайн Lisp, гомоиконность, образовательная ценность, интерактивная разработка и постоянная активность в Common Lisp, Scheme, Emacs Lisp и более новых диалектах. Преобладающее мнение однозначно положительное, хотя в некоторых дискуссиях признаётся нишевый статус Lisp, ограничения в инструментарии и неуверенность в более широком внедрении.

![](https://habrastorage.org/webt/cb/6b/bc/cb6bbc51c74ef0313947b9e5c879fd6a.png)

А также заметно, что большое количество постов — рассказы о том, почему именно Lisp.

![](https://habrastorage.org/webt/75/3a/66/753a6658544fd37600d36f2577e1a276.png)

## Hacking and Security Research

Тема взломов и исследований в сфере безопасности всегда актуальна, так что и обнаружился кластер, где рассказывается об уязвимостях, социальной инженерии, вредоносных программах. Сообщество восхищено творческим подходом и образовательной ценностью исследований в области безопасности, но по-прежнему обеспокоено слабой защитой, возможностью использования, плохим отношением к исследователям и ущербом, причиняемым реальными атаками.

![](https://habrastorage.org/webt/27/bc/dc/27bcdc7529b40034498ee6a4ab675ed8.png)

# Заключение

Анализируя два года постов на Hacker News, можно понять основные устойчивые тренды в мире ИТ. AI — главная тема, но дискуссия вокруг него расколота. Сообщество не может прийти к единому мнению: одни видят в AI новый инструмент, другие — угрозу или шум, а третьи против и первых, и вторых. А Open Source переходит в фазу «взросления». Вопросы лицензирования, финансирования и борьбы с нейрослопом заставляют сообщество адаптироваться.

Настроение, кстати, колеблется. Оптимизм сменяется тревогой, особенно в контексте AI и экономической нестабильности. Hacker News для многих публикуется в рабочие дни, что можно было увидеть на графиках.

Этот датасет можно использовать как для анализа, так и для собственных нужд: чтения интересных статей, поиска вдохновения по теме, ресерча HN в поиске интересных постов, идей или проектов.

Исходный код доступен в [моём репозитории](https://github.com/alexeev-prog/hn-sentiment-analysis). А онлайн-карта доступна по [этой ссылке](https://alexeev-prog.github.io/hn-sentiment-analysis/).
