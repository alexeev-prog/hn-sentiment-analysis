# Я проанализировал 100 000 постов с Hacker News и вот что хочу сказать
Hacker News — это пульс мирового технологического сообщества. Тысячи разработчиков, основателей стартапов, инженеров из крупных компаний ежедневно делятся здесь инсайтами, проектами, новостями и мемами. Но есть проблема: качественный контент тонет в шуме, или интересные тебе темы тонут в неинтересных тебе темах.

Но из-за большого потока постов, качественные статьи по нужной теме теряются среди других, и если нужно найти какие-то системные тренды, увидеть как меняется отношения к Rust, AI или Open Source, а не просто "почитать что-то интересное" - это сделать сложно. Можно да, скопировать кучу ссылок и отправить в LLM, попросив отсортировать, но это занимает время и не так удобно, если нужно проанализировать большое количество тем.

И тут мне пришла идея - создать программу на Python, которая будет парсить через официальный API Hacker News, строить эмбеддинги постов с комментариями, кластеризовать посты и при помощи LLM суммаризировать этот кластер по ссылкам.

Сказано - сделано. Я смог спарсить 100 000 постов с Hacker News за последние 693 дня, получив 1031 кластеров, 10.4 миллиона апвоутов, 27 тысячи авторов и всего 0.113% выбросов. В этой статье мы разберем, какие тренды были активны в мировом IT-сообществе последние два года.

---

Итак, однажды, в процессе скроллинга Hacker News в поиске вдохновения и интересных статей, я задумался, у hacker news есть целых два API (firebase и algolia), так почему бы не придумать какой нибудь проект с использованием API? Но обычный постинг в условный телеграм-канал или просто приложение, показалось мне скучным. Немного подумав, я пришел к идее, а почему бы не автоматизировать и не проанализировать посты на hacker news? Это и идеи для статей, и возможность почитать и проанализировать мнения по разным темам, найти интересующие вещи.

Я начал думать, какие технологии пригодятся. Просто кидать LLM ссылки и просить выделить темы - скучно. Я решил пойти путем другим, через создание ETL (Extract-Transform-Load) пайплайна. Сначала мы парсим сами статьи, и если позволяет rate-limit - то и 3-5 комментариев к ним. После мы формируем из них объекты данных, которые в последующем через трансформер all-MiniLM-L6-v2 преобразуем в эмбеддинги. Эти самые эмбеддинги кластеризируются через HDBSCAN, а затем LLM анализирует и создает заголовок и саммари для каждого из кластеров. И в конце - создаем json и html файл, графики различные.

Конечно, в процессе разработки я сталкивался с проблемами, например оптимизация запросов к API, настройка кластеризации с маленьким выбросом (т.е. некластерезированными статьями), работа с LLM через API, конфигурация и сам пайплайн. Но итоговый результат того стоил, за несколько минут мы получаем двухлетний срез всех мнений в ИТ-сообществе.

Сам исходный код программы, которую я разработал для сбора данных, вы можете [прочитать в моем репозитории](https://github.com/alexeev-prog/hn-sentiment-analysis). Перед тем, как начать разбор, хочу сказать, что я также сделал онлайн карту историй, доступную [по этой ссылке](https://alexeev-prog.github.io/hn-sentiment-analysis/). Для анализа требуется файл кластеризации, вы можете сгенерировать как и сами через мой репозиторий, так и скачать [готовый файл кластеризации hn_cluters.json](https://github.com/alexeev-prog/hn-sentiment-analysis/releases/tag/1.0).

В этой статье я разберу какую работу я проделал, и также проанализируем графики, кластеры и статистику, и узнаем, что было на уме в IT-сфере последние годы.

# Архитектура решения
Я решил создать ETL-пайплайн для того, чтобы грамотно собрать все данные и кластеризировать все. Пайплайн состоит из четырёх логических этапов: извлечение, трансформация, кластеризация и суммаризация.

Начнем с этапа извлечения данных. Тут довольно интересно, hacker news имеет два официальных API - [Firebase Hacker News API](https://github.com/HackerNews/API) и [Algolia API](https://hn.algolia.com/api). Firebase дает прямой доступ к сырым данным в реальном времени. Отлично подходит для мониторинга новых постов, но для анализа вглубь, увы, неудобен: нужно проходиться по всем ID, делать сотни запросов к /item/, фильтрация и сортировка — на стороне клиента. А вот Algolia удобен и своим поисковым движком, и готовыми индексами. Он позволяет фильтровать, сортировать, ограничивать по дате и скору, и поэтому я выбрал его.

Но у algolia есть лимиты и ограничения, например не более 1000 постов за раз.

У меня в коде, в части связанной с фильтрацией, есть параметр `search_by_date`. Если он False, то мы используем API-эндпоинт `https://hn.algolia.com/api/v1/search`, и он отдает по релевантности. А по умолчанию он равен True, и используется уже API-эндпоинт `https://hn.algolia.com/api/v1/search_by_date`. Вместо того чтобы запрашивать страницы подряд, я использую параметр `created_at_i<{timestamp}` и двигаюсь назад во времени. Механика простая: запросили 100 постов с датой меньше текущей, взяли минимальную дату из ответа, подставили ее в следующий запрос. Так можно собрать хоть 100 000 постов — пока не упремся в max_pages или не кончатся данные.

Также есть и фильтрация, через QueryParams, который строит словарь для передачи в API.

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

Кроме того, есть `AlgoliaStoryFilter`, это абстрактный класс для удобного создания различных фильтров:

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

Все это работает уже через наши DTO: Story и Comment:

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

Следующий этап - трансформация, то есть превращение данных в векторы, эмбеддинги.

Текст эмбеддинга формируется просто: заголовок повторяется три раза (самый сильный сигнал, чтобы комментарии не могли увести куда то в сторону). Далее добавляем комментарии с обрезкой до 350 символов.

Как трансформер я использую all-MiniLM-L6-v2. Модель быстрая и качественная, выдает нормализованные эмбеддинги (normalize_embeddings=True), что критично для кластеризации через косинусное расстояние. Да, можно воспользоваться более качественными решениями или API, но они медленные, а в случае огромной массы историй скорость важна.

Следующий этап - кластеризация. Тут выбор был между HDBSCAN и K-means. Я выбрал HDBSCAN, так как у KMeans нужно было знать итоговое количество кластеров, а у нас неизвестно. А также он умеет маркировать шум (noise) — посты, которые не вписались ни в один кластер.

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

Редукция до 12 измерений явлется компромиссом: сохраняет структуру данных, но ускоряет кластеризацию. В коде также сохраняются первые две компоненты `(points[:, :2])` для визуализации на карте.

А теперь к обработке выбросов. HDBSCAN обычно оставляет 10–30% точек как шум. Моя задача — минимизировать потери и и стараться прийти к тому, чтобы все посты были распределены по кластерам (и чтобы кластеров было достаточно много). В коде реализован алгоритм присвоения выбросов:

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

Порог 0.45 отсекает точки, которые слишком далеки от любого кластера. В результате из 100 000 постов только 0.11% остались некластеризованными. Это говорит о том, что UMAP+HDBSCAN с правильными параметрами (плюс релевантная выборка) отлично разделяет темы.

---

# КЛЮЧЕВЫЕ СЛОВА

```
hacker news	406
айти	363 004
тренды	1 547 176
it тренды	846
ии	9 615 674
```
