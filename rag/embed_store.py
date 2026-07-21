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
from sentence_transformers import SentenceTransformer
import faiss

from .ingest import Chunk


class VectorStore:
    def __init__(self):
        """
        SentenceTransformers embedding model and FAISS vector index.
        Uses 'all-MiniLM-L6-v2' (384 dimensions), which is extremely fast and high-quality.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.index = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        """Create the FAISS index for all anime documents."""
        self.chunks = list(chunks)
        if not chunks:
            self.index = faiss.IndexFlatIP(self.dimension)
            return

        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        
        # L2-normalize embeddings so that inner product equals cosine similarity
        faiss.normalize_L2(embeddings)

        # Initialize FAISS Index (IndexFlatIP for normalized Inner Product)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype('float32'))

        print(f"Indexed {len(chunks)} anime documents using SentenceTransformer and FAISS.")

    def add_chunks(self, new_chunks: List[Chunk]) -> None:
        """Dynamically add new chunks to the in-memory vector database."""
        if not new_chunks:
            return

        texts = [chunk.text for chunk in new_chunks]
        
        # Generate embeddings for new chunks
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(new_chunks)
        print(f"Added {len(new_chunks)} new chunks to the index. Total: {len(self.chunks)} chunks.")

    def query(
        self,
        query_text: str,
        top_k: int = 3
    ) -> List[Tuple[Chunk, float]]:
        """
        Return the top matching anime documents.
        """
        if self.index is None or not self.chunks:
            return []

        # Encode and normalize query
        query_vec = self.model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)

        # Handle top_k larger than number of chunks
        actual_k = min(top_k, len(self.chunks))
        if actual_k <= 0:
            return []

        # Query index
        scores, indices = self.index.search(query_vec.astype('float32'), actual_k)

        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS padding
                continue
            results.append((self.chunks[idx], float(score)))

        return results