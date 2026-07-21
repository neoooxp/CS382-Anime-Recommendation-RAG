"""
Vector store: turn chunks into vectors and support similarity search.

Current implementation:
- TF-IDF
- Cosine similarity
- Optimized for ~1000 anime documents

Possible future upgrades:
- SentenceTransformers embeddings
- FAISS / Chroma vector database
"""

from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingest import Chunk


class VectorStore:
    def __init__(self):
        """
        TF-IDF vectorizer.

        ngram_range=(1,2)
            Uses both single words and word pairs.

        stop_words="english"
            Removes common English filler words.

        lowercase=True
            Makes searching case-insensitive.
        """
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2)
        )

        self.matrix = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        """Create the TF-IDF matrix for all anime documents."""

        self.chunks = chunks

        texts = [chunk.text for chunk in chunks]

        self.matrix = self.vectorizer.fit_transform(texts)

        print(f"Indexed {len(chunks)} anime documents.")

    def query(
        self,
        query_text: str,
        top_k: int = 3
    ) -> List[Tuple[Chunk, float]]:
        """
        Return the top matching anime documents.
        """

        if self.matrix is None:
            raise RuntimeError(
                "VectorStore.build() must be called before query()."
            )

        query_vec = self.vectorizer.transform([query_text])

        scores = cosine_similarity(
            query_vec,
            self.matrix
        ).flatten()

        ranked = np.argsort(scores)[::-1][:top_k]

        return [
            (self.chunks[i], float(scores[i]))
            for i in ranked
        ]