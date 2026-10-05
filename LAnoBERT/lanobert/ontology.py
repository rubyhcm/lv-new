"""Small local ontology retriever used before introducing a vector database."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class OntologyDocument:
    id: str
    text: str
    source: str = "unknown"


class TfidfOntologyRetriever:
    """Deterministic top-k text retrieval with cosine similarity."""

    def __init__(self, documents: Sequence[OntologyDocument]):
        if not documents:
            raise ValueError("at least one ontology document is required")
        self.documents = list(documents)
        self.vectorizer = TfidfVectorizer(lowercase=True)
        self.matrix = self.vectorizer.fit_transform(doc.text for doc in self.documents)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[OntologyDocument, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.documents[index], float(scores[index])) for index in order]
