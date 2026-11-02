"""Local, extractive search with explicit provenance and status boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence


MISSING_INFORMATION = "В утверждённом каноне DENGO эта информация не указана"
NOT_FOUND_MESSAGE = (
    "Не найдено поиском в индексированных текстовых источниках. "
    "Это не подтверждает отсутствие сведений во всех материалах канона."
)
STATUS_NAMES = ("КАНОН", "ПРЕДЛОЖЕНИЕ", "НЕ РЕШЕНО", "СВЕРИТЬ")
REVIEW_STATUSES = ("ПРЕДЛОЖЕНИЕ", "НЕ РЕШЕНО", "СВЕРИТЬ", "НЕОПРЕДЕЛЁН", "БЕЗ СТАТУСА")
CATEGORY_BY_DIR = {
    "characters": "персонажи",
    "countries": "государства",
    "monsters": "чудовища",
    "seals": "печати",
    "story": "история",
    "world": "устройство мира",
}
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
_MARKER_RE = re.compile(r"\[([^\]]+)\]")
_CLAUSE_RE = re.compile(r"(?<=[.;!?])\s+(?=[А-ЯЁA-Z])")

_STOP_WORDS = {
    "а", "без", "бы", "был", "была", "были", "быть", "в", "во", "вот",
    "все", "вы", "где", "да", "для", "до", "его", "ее", "её", "если", "есть",
    "за", "и", "из", "или", "им", "их", "как", "кто", "ли", "между", "мне",
    "на", "над", "не", "него", "нее", "неё", "нет", "ни", "но", "о", "об",
    "он", "она", "они", "оно", "от", "по", "под", "при", "с", "со", "так",
    "там", "то", "у", "что", "это", "этот", "эта", "эти", "я", "расскажи",
    "скажи", "известно", "указано", "напиши", "кратко", "перечисли", "существуют",
    "существует", "такой", "такая", "такие", "какие", "какая", "какой", "каково",
    "какова", "каков", "сколько", "есть", "ещё", "еще", "утверждены", "утверждено",
}

_SUFFIXES = tuple(sorted({
    "иями", "ями", "ами", "ого", "ему", "ому", "ыми", "ими", "иях", "иях",
    "ов", "ев", "ей", "ий", "ый", "ая", "яя", "ое", "ее", "ые", "ие",
    "ам", "ям", "ах", "ях", "ом", "ем", "ой", "ей", "ую", "юю", "а", "я",
    "ы", "и", "у", "ю", "е", "о", "ь", "м", "х", "й",
}, key=len, reverse=True))

_FIELD_TERMS = {
    "длина": {"длин", "длина", "размер"},
    "ширина": {"ширин"},
    "высота": {"высот"},
    "уровень": {"уров", "уровн"},
    "причина": {"причин"},
}
_TOPIC_TERMS = {
    "государства": {"государств", "стран", "географ", "столиц"},
    "персонажи": {"персонаж", "геро", "биограф"},
    "чудовища": {"чудовищ", "монстр", "дракон", "подземел"},
    "печати": {"печат", "антипечат", "джи", "клан"},
    "история": {"войн", "истор", "хронолог", "дат", "причин"},
    "устройство мира": {"мир", "бог", "создал", "мифолог", "донзи", "хони"},
}

_CHARACTER_HEADINGS = {
    "АРТО РЕН": ("Арто", {"арто", "арто рен"}),
    "РЕН КУРОСАКИ": ("Рен Куросаки", {"рен куросаки"}),
    "БРОК ШОН": ("Брок Шон", {"брок", "брок шон"}),
    "СОРА МИНАДЗУКИ": ("Сора Минадзуки", {"сора", "сора минадзуки"}),
}


@dataclass(frozen=True)
class CanonFragment:
    status: str
    source: str
    start_line: int
    end_line: int
    text: str
    category: str = "устройство мира"
    entity: str | None = None


@dataclass(frozen=True)
class Finding:
    state: str
    fragments: tuple[CanonFragment, ...] = ()


def _marker_status(line: str) -> str | None:
    found: set[str] = set()
    for marker in _MARKER_RE.findall(line):
        value = marker.strip().upper()
        matches = [
            name for name in STATUS_NAMES
            if re.search(r"(?<![А-ЯЁA-Z])" + re.escape(name) + r"(?![А-ЯЁA-Z])", value)
        ]
        if len(matches) != 1:
            return "НЕОПРЕДЕЛЁН"
        found.add(matches[0])
    if len(found) > 1:
        return "НЕОПРЕДЕЛЁН"
    return next(iter(found), None)


def _normalise(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.casefold())).strip()


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]
    return token


def _tokens(text: str) -> set[str]:
    return {
        _stem(token.casefold())
        for token in _TOKEN_RE.findall(text)
        if token.casefold() not in _STOP_WORDS and len(token) > 1
    }


def _category_for_source(source: str) -> str:
    first = Path(source).parts[0].casefold() if Path(source).parts else ""
    return CATEGORY_BY_DIR.get(first, "устройство мира")


def _is_heading(line: str) -> bool:
    text = line.strip()
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and text == text.upper() and not text.startswith(("-", "•"))


def _local_status(text: str, inherited: str) -> str:
    folded = text.casefold()
    if any(term in folded for term in ("не считать каноном", "не является утверждённым каноном", "не считать окончательным каноном", "проектные данные", "предложен")):
        return "ПРЕДЛОЖЕНИЕ"
    if any(term in folded for term in ("требует сверки", "сверить с", "нужно сверить", "не подтвержден", "не подтверждён")):
        return "СВЕРИТЬ"
    if any(term in folded for term in ("не решено", "не восстановлен", "не восстановлены", "ещё не утвержден", "еще не утвержден", "не указана точная", "не указано полное")):
        return "НЕ РЕШЕНО"
    return inherited


def _split_local_clauses(text: str) -> list[str]:
    clauses = [part.strip() for part in _CLAUSE_RE.split(text.strip()) if part.strip()]
    return clauses or [text.strip()]


def _infer_entity(category: str, text: str, scope_entity: str | None) -> str | None:
    if scope_entity:
        return scope_entity
    folded = text.casefold()
    if category == "чудовища" and "ханзо" in folded:
        return "Ханзо"
    if category == "государства":
        for name in ("Золами", "Бурлин", "Балбул", "Реян", "Зовлил"):
            if name.casefold() in folded:
                return name
    return None


def _make_fragments_for_line(
    text: str,
    status: str,
    source: str,
    line_number: int,
    category: str,
    scope_entity: str | None,
) -> list[CanonFragment]:
    result = []
    for clause in _split_local_clauses(text):
        clause_status = _local_status(clause, status)
        result.append(CanonFragment(
            status=clause_status,
            source=source,
            start_line=line_number,
            end_line=line_number,
            text=clause,
            category=category,
            entity=_infer_entity(category, clause, scope_entity),
        ))
    return result


def parse_canon_text(text: str, source: str = "<memory>") -> list[CanonFragment]:
    """Parse each list item with exact line provenance and conservative status scope."""
    category = _category_for_source(source)
    status = "БЕЗ СТАТУСА"
    scope_entity: str | None = None
    fragments: list[CanonFragment] = []

    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        marker_status = _marker_status(line)
        if marker_status is not None:
            status = marker_status
            # Markers are headings in this corpus; a trailing claim on the same
            # line is still parsed and retains that exact source line.
            line = _MARKER_RE.sub("", line).strip()
            if not line:
                continue

        if _is_heading(line):
            # Unlabelled section headings terminate the preceding status scope.
            if marker_status is None:
                status = "БЕЗ СТАТУСА"
            if category == "персонажи":
                heading = line.upper()
                scope_entity = _CHARACTER_HEADINGS.get(heading, (None, set()))[0]
            elif category == "чудовища":
                scope_entity = "Ханзо" if "ХАНЗО" in line.upper() else None
            continue

        if marker_status == "НЕОПРЕДЕЛЁН":
            status = "НЕОПРЕДЕЛЁН"

        # A bullet is a source-level claim. Wrapped non-bullet lines remain
        # separate claims to avoid losing line provenance or merging statuses.
        clean = re.sub(r"^[\-•]\s*", "", line).strip()
        if clean:
            fragments.extend(_make_fragments_for_line(
                clean, status, source, number, category, scope_entity
            ))
    return fragments


def load_fragments(canon_dir: Path) -> list[CanonFragment]:
    fragments: list[CanonFragment] = []
    if not canon_dir.exists():
        return fragments
    for path in sorted(canon_dir.rglob("*.txt")):
        if path.name == "README_RU.txt":
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        fragments.extend(parse_canon_text(content, str(path.relative_to(canon_dir))))
    return fragments


def _topic_categories(question: str) -> set[str]:
    terms = _tokens(question)
    categories = set()
    for category, markers in _TOPIC_TERMS.items():
        if any(any(term.startswith(marker) or marker.startswith(term) for term in terms) for marker in markers):
            categories.add(category)
    # Creature or named-monster requests use the monster source even when the
    # query is phrased as a person question ("Кто такой Ханзо?").
    if "ханзо" in _normalise(question):
        categories.add("чудовища")
    if len(categories) > 1:
        categories.discard("устройство мира")
    return categories


def _resolve_entity(question: str, fragments: Sequence[CanonFragment]) -> tuple[str | None, list[str]]:
    normalized = _normalise(question)
    query_tokens = _tokens(question)
    if "арто рен" in normalized:
        return "Арто", []
    if re.search(r"(?<![а-яёa-z])арто(?![а-яёa-z])", normalized):
        return "Арто", []
    if "куросаки" in normalized and "рен" in query_tokens:
        return "Рен Куросаки", []
    for full_name, (entity, aliases) in _CHARACTER_HEADINGS.items():
        if full_name == "АРТО РЕН":
            continue
        if any(alias in normalized for alias in aliases):
            return entity, []
    if "ханзо" in normalized:
        return "Ханзо", []
    for name in ("Золами", "Бурлин", "Балбул", "Реян", "Зовлил"):
        if name.casefold() in normalized:
            return name, []
    if "рен" in query_tokens:
        if any(f.category == "печати" and "Рен" in f.text for f in fragments):
            return None, ["Рен Куросаки", "Рен — одно из имён глав кланов Печатей"]
        return "Рен Куросаки", []
    return None, []


def _requested_fields(question: str) -> list[str]:
    terms = _tokens(question)
    requested = []
    for field, markers in _FIELD_TERMS.items():
        if any(any(term.startswith(marker) or marker.startswith(term) for term in terms) for marker in markers):
            requested.append(field)
    return requested


def assess_finding(fragments: Sequence[CanonFragment], exhaustive_review: bool = False) -> Finding:
    """Keep search failure distinct from a genuinely exhaustive negative review."""
    if fragments:
        return Finding("FOUND", tuple(fragments))
    if exhaustive_review:
        return Finding("ABSENT_AFTER_REVIEW")
    return Finding("NOT_FOUND_BY_SEARCH")


def _field_matches(field: str, fragment: CanonFragment) -> bool:
    text = fragment.text.casefold()
    if field == "длина":
        return "длин" in text or ("метр" in text and fragment.entity == "Ханзо")
    if field == "ширина":
        return "ширин" in text
    if field == "высота":
        return "высот" in text
    if field == "уровень":
        return "уров" in text
    if field == "причина":
        return "причин" in text and not any(
            phrase in text
            for phrase in ("нельзя менять", "нельзя придумывать", "не придумывать заново")
        )
    return False


def _all_canon(fragments: Iterable[CanonFragment]) -> list[CanonFragment]:
    return [fragment for fragment in fragments if fragment.status == "КАНОН"]


def relevant_canon(question: str, fragments: Iterable[CanonFragment], limit: int = 6) -> list[CanonFragment]:
    """Select source evidence with topic, entity, and morphological token checks."""
    all_fragments = list(fragments)
    canon = _all_canon(all_fragments)
    entity, ambiguous = _resolve_entity(question, all_fragments)
    if ambiguous:
        return []
    categories = _topic_categories(question)
    if entity:
        selected = [fragment for fragment in canon if fragment.entity == entity]
        # Explicit query aliases can target source entities without a section
        # heading, but never pull another character's scoped section.
        if entity in {"Арто", "Рен Куросаки", "Брок Шон", "Сора Минадзуки"}:
            selected = [fragment for fragment in selected if fragment.category == "персонажи"]
        if entity in {"Золами", "Бурлин", "Балбул", "Реян", "Зовлил"}:
            selected = [fragment for fragment in canon if fragment.entity == entity and fragment.category == "государства"]
        if entity == "Ханзо":
            selected = [fragment for fragment in canon if fragment.category == "чудовища" and fragment.entity == "Ханзо"]
        return selected
    if categories:
        allowed = [fragment for fragment in canon if fragment.category in categories]
        query_tokens = _tokens(question)
        # Broad category questions should enumerate that category, including
        # facts whose wording differs from the query (e.g. "государства").
        raw_tokens = {token.casefold() for token in _TOKEN_RE.findall(question)}
        broad = (
            "государства" in categories
            and any(token.startswith(("какие", "какой", "список", "перечисл", "существ")) for token in raw_tokens)
        )
        if broad:
            return allowed[:max(limit, 50)]
        scored = []
        for index, fragment in enumerate(allowed):
            overlap = len(query_tokens & _tokens(fragment.text))
            if overlap:
                scored.append((overlap, -index, fragment))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:limit]]
    query_tokens = _tokens(question)
    scored = []
    for index, fragment in enumerate(canon):
        overlap = len(query_tokens & _tokens(fragment.text))
        if overlap:
            scored.append((overlap, -index, fragment))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:limit]]


def format_evidence(fragments: Iterable[CanonFragment]) -> str:
    blocks = []
    for fragment in fragments:
        blocks.append(
            f"[{fragment.status}] Тема: {fragment.category}; субъект: {fragment.entity or 'не выделен'}\n"
            f"Источник: {fragment.source}, строки {fragment.start_line}-{fragment.end_line}\n"
            f"{fragment.text}"
        )
    return "\n\n".join(blocks)


def _format_finding(field: str, finding: Finding) -> str:
    if finding.state == "FOUND":
        return f"Параметр «{field}»: найдено подтверждение.\n{format_evidence(finding.fragments)}"
    if finding.state == "ABSENT_AFTER_REVIEW":
        return (
            f"Параметр «{field}»: проверено во всех материалах указанного полного охвата; "
            f"{MISSING_INFORMATION}."
        )
    return f"Параметр «{field}»: {NOT_FOUND_MESSAGE}"


def format_finding(field: str, finding: Finding) -> str:
    return _format_finding(field, finding)


def answer_from_sources(question: str, fragments: Iterable[CanonFragment]) -> tuple[str, list[CanonFragment]]:
    corpus = list(fragments)
    entity, ambiguous = _resolve_entity(question, corpus)
    if ambiguous:
        options = " или ".join(ambiguous)
        return f"Уточните, кого вы имеете в виду: {options}?", []

    fields = _requested_fields(question)
    if fields:
        categories = _topic_categories(question)
        subject_fragments = [
            fragment for fragment in _all_canon(corpus)
            if (entity is None or fragment.entity == entity)
            and (not categories or fragment.category in categories)
        ]
        sections = []
        used: list[CanonFragment] = []
        for field in fields:
            matches = [fragment for fragment in subject_fragments if _field_matches(field, fragment)]
            finding = assess_finding(matches, exhaustive_review=False)
            sections.append(_format_finding(field, finding))
            used.extend(matches)
        return "\n\n".join(sections), used

    raw_stems = {_stem(token.casefold()) for token in _TOKEN_RE.findall(question)}
    if any(term.startswith(prefix) for term in raw_stems for prefix in ("утвержд", "нереш", "предлож", "свер")):
        return (
            "Рабочие сведения не включаются в ответы канона. "
            "Для их отдельного просмотра используйте GET /review?status=ПРЕДЛОЖЕНИЕ, "
            "GET /review?status=НЕ%20РЕШЕНО, GET /review?status=СВЕРИТЬ "
            "или безопасные статусы НЕОПРЕДЕЛЁН и БЕЗ%20СТАТУСА.",
            [],
        )

    selected = relevant_canon(question, corpus)
    if not selected:
        return NOT_FOUND_MESSAGE, []
    return format_evidence(selected), selected


def review_fragments(status: str, fragments: Iterable[CanonFragment]) -> list[CanonFragment]:
    normalized = status.strip().upper()
    if normalized not in REVIEW_STATUSES:
        raise ValueError(f"Статус должен быть одним из: {', '.join(REVIEW_STATUSES)}")
    return [fragment for fragment in fragments if fragment.status == normalized]
