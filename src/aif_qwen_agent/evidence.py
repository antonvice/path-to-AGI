import re
from pathlib import PurePosixPath

_WORD = re.compile(r"[a-z0-9]+")
_READ_INTENT = re.compile(
    r"\b(?:contents?|read|recorded|specif(?:y|ies|ied)|summari[sz]e)\b",
    re.IGNORECASE,
)
_NEGATED_READ = re.compile(r"\b(?:do not|don['’]t|never)\s+read\b", re.IGNORECASE)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "does",
    "in",
    "is",
    "it",
    "of",
    "or",
    "report",
    "specify",
    "the",
    "value",
    "what",
    "with",
}
_MAX_EXCERPT_CHARS = 512
_FULL_CONTENT_THRESHOLD = 96


def extract_explicit_file_path(text: str) -> str | None:
    if not _READ_INTENT.search(text) or _NEGATED_READ.search(text):
        return None
    candidates: list[str] = []
    for raw in text.split():
        candidate = raw.strip("\"'`()[]{}<>").rstrip(",;:!?.").rstrip("\"'`()[]{}<>")
        path = PurePosixPath(candidate)
        if (
            not candidate
            or "://" in candidate
            or "@" in candidate
            or path.is_absolute()
            or not path.suffix
        ):
            continue
        candidates.append(candidate)
    unique = tuple(dict.fromkeys(candidates))
    return unique[0] if len(unique) == 1 else None


def _terms(text: str) -> set[str]:
    return {term for term in _WORD.findall(text.casefold()) if term not in _STOP_WORDS}


def project_evidence(task_text: str, content: str) -> str:
    if len(content) <= _FULL_CONTENT_THRESHOLD:
        return content.rstrip("\n")
    path = extract_explicit_file_path(task_text)
    query_terms = _terms(task_text.replace(path, " ") if path is not None else task_text)
    lines = content.splitlines()
    if not lines:
        return content[:_MAX_EXCERPT_CHARS]
    scores = [sum(map(len, query_terms & _terms(line))) for line in lines]
    best = scores.index(max(scores)) if any(scores) else 0
    return lines[best][:_MAX_EXCERPT_CHARS]
