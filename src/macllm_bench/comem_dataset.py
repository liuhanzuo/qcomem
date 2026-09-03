from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    relevant_document_ids: tuple[str, ...]
    selected_document_ids: tuple[str, ...]
    expected_answer: str | None = None


@dataclass(frozen=True)
class MultiDocumentDataset:
    name: str
    documents: tuple[Document, ...]
    queries: tuple[Query, ...]

    @property
    def documents_by_id(self) -> dict[str, Document]:
        return {document.id: document for document in self.documents}


def _required_text(row: dict[str, Any], field: str, kind: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{kind}.{field} must be a non-empty string")
    return value.strip()


def _id_list(row: dict[str, Any], field: str, kind: str) -> tuple[str, ...]:
    value = row.get(field, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{kind}.{field} must be a list of document IDs")
    if len(value) != len(set(value)):
        raise ValueError(f"{kind}.{field} contains duplicate document IDs")
    return tuple(value)


def load_dataset(path: Path | str) -> MultiDocumentDataset:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("dataset root must be a JSON object")
    name = _required_text(payload, "name", "dataset")
    raw_documents = payload.get("documents")
    raw_queries = payload.get("queries")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ValueError("dataset.documents must be a non-empty list")
    if not isinstance(raw_queries, list) or not raw_queries:
        raise ValueError("dataset.queries must be a non-empty list")

    documents = []
    for row in raw_documents:
        if not isinstance(row, dict):
            raise ValueError("each document must be an object")
        documents.append(
            Document(
                id=_required_text(row, "id", "document"),
                title=_required_text(row, "title", "document"),
                text=_required_text(row, "text", "document"),
            )
        )
    document_ids = [document.id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("document IDs must be unique")
    known_ids = set(document_ids)

    queries = []
    for row in raw_queries:
        if not isinstance(row, dict):
            raise ValueError("each query must be an object")
        relevant = _id_list(row, "relevant_document_ids", "query")
        selected = _id_list(row, "selected_document_ids", "query")
        unknown = (set(relevant) | set(selected)) - known_ids
        if unknown:
            raise ValueError(f"query references unknown documents: {sorted(unknown)}")
        answer = row.get("expected_answer")
        if answer is not None and (not isinstance(answer, str) or not answer.strip()):
            raise ValueError("query.expected_answer must be null or non-empty text")
        queries.append(
            Query(
                id=_required_text(row, "id", "query"),
                text=_required_text(row, "query", "query"),
                relevant_document_ids=relevant,
                selected_document_ids=selected,
                expected_answer=answer.strip() if answer else None,
            )
        )
    query_ids = [query.id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique")
    return MultiDocumentDataset(name, tuple(documents), tuple(queries))


def render_document(document: Document) -> str:
    return (
        f"Document ID: {document.id}\n"
        f"Title: {document.title}\n"
        f"Content: {document.text}\n"
    )


def render_query_prefix(query: Query) -> str:
    return f"Question: {query.text}\nAnswer:\n"


def _terms(text: str) -> list[str]:
    # Individual CJK characters make the small built-in selector usable without
    # adding a language-specific tokenizer dependency. This is a selector
    # baseline, not a claim about production Chinese retrieval quality.
    return re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", text.lower())


class BM25Selector:
    """Small deterministic CPU BM25 baseline for selector sensitivity tests."""

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25 requires at least one document")
        self.documents = tuple(documents)
        self.k1 = k1
        self.b = b
        self.terms = [
            _terms(f"{document.title} {document.text}") for document in documents
        ]
        self.lengths = [len(terms) for terms in self.terms]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.document_frequency: dict[str, int] = {}
        for terms in self.terms:
            for term in set(terms):
                self.document_frequency[term] = (
                    self.document_frequency.get(term, 0) + 1
                )

    def score(self, query: str) -> list[tuple[str, float]]:
        query_terms = _terms(query)
        count = len(self.documents)
        rows = []
        for document, terms, length in zip(
            self.documents, self.terms, self.lengths, strict=True
        ):
            frequencies: dict[str, int] = {}
            for term in terms:
                frequencies[term] = frequencies.get(term, 0) + 1
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = self.document_frequency.get(term, 0)
                inverse_frequency = math.log(
                    1 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                normalization = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.average_length, 1)
                )
                score += inverse_frequency * frequency * (self.k1 + 1) / normalization
            rows.append((document.id, score))
        return sorted(rows, key=lambda row: (-row[1], row[0]))

    def select(self, query: str, top_k: int) -> list[str]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        return [document_id for document_id, _ in self.score(query)[:top_k]]


def evidence_recall(selected: Sequence[str], relevant: Sequence[str]) -> float | None:
    if not relevant:
        return None
    return len(set(selected) & set(relevant)) / len(set(relevant))
