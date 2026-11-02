from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from canon_guard import (
    answer_from_sources,
    format_evidence,
    load_fragments,
    review_fragments,
)


app = FastAPI(title="DENGO AI STUDIO")

BASE_DIR = Path(__file__).resolve().parent / "data" / "knowledge"
RULES_FILE = BASE_DIR / "system_rules.txt"
CANON_DIR = BASE_DIR / "canon"


class ChatRequest(BaseModel):
    message: str


def load_rules():
    if RULES_FILE.exists():
        return RULES_FILE.read_text(encoding="utf-8")
    return "Правила проекта не найдены."


def load_canon_files():
    """Keep the existing knowledge endpoint, with explicit read failures omitted."""
    files = []
    if not CANON_DIR.exists():
        return files
    for path in sorted(CANON_DIR.rglob("*.txt")):
        if path.name == "README_RU.txt":
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        files.append({"name": str(path.relative_to(CANON_DIR)), "content": content})
    return files


def find_relevant_canon(question):
    answer, _ = answer_from_sources(question, load_fragments(CANON_DIR))
    return answer


@app.get("/")
def home():
    return {"name": "DENGO AI STUDIO", "status": "Online"}


@app.get("/rules")
def get_rules():
    return {"status": "Rules loaded", "rules": load_rules()}


@app.get("/knowledge")
def get_knowledge():
    files = load_canon_files()
    fragments = load_fragments(CANON_DIR)
    status_counts = {}
    category_counts = {}
    for fragment in fragments:
        status_counts[fragment.status] = status_counts.get(fragment.status, 0) + 1
        category_counts[fragment.category] = category_counts.get(fragment.category, 0) + 1
    return {
        "status": "Knowledge loaded",
        "files_loaded": len(files),
        "files": [file["name"] for file in files],
        "fragments_by_status": status_counts,
        "fragments_by_category": category_counts,
        "indexed_formats": [".txt"],
        "not_indexed": ["Dengo_Full_World_Archive_original.pdf", "DENGO_CANON_PACKAGE.zip"],
    }


@app.post("/chat")
def chat(request: ChatRequest):
    fragments = load_fragments(CANON_DIR)
    answer, selected = answer_from_sources(request.message, fragments)
    if answer.startswith("Уточните, кого"):
        response_status = "clarification_required"
    elif answer.startswith("Не найдено поиском"):
        response_status = "not_found_by_search"
    elif "Не найдено поиском в индексированных текстовых источниках" in answer and selected:
        response_status = "partial"
    else:
        response_status = "success"
    return {
        "status": response_status,
        "model": "extractive-local-guard",
        "canon_used": format_evidence(selected),
        "answer": answer,
    }


@app.get("/review")
def review(status: str = Query(..., description="ПРЕДЛОЖЕНИЕ, НЕ РЕШЕНО, СВЕРИТЬ, НЕОПРЕДЕЛЁН или БЕЗ СТАТУСА")):
    try:
        selected = review_fragments(status, load_fragments(CANON_DIR))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "status": status.strip().upper(),
        "count": len(selected),
        "items": [
            {
                "status": fragment.status,
                "category": fragment.category,
                "entity": fragment.entity,
                "source": fragment.source,
                "start_line": fragment.start_line,
                "end_line": fragment.end_line,
                "text": fragment.text,
            }
            for fragment in selected
        ],
    }
