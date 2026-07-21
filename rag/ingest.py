"""
Ingestion: load anime documents and split them into individual anime chunks.

Changes for this project:
- Load plain text files from the dataset
- Split documents using <ANIME> ... </ANIME> tags
- Extract useful metadata (UID, title, rank, genre, source file)
- One anime = one embedding chunk
"""

import os
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_documents(folder: str) -> List[dict]:
    """
    Load every .txt file into memory.
    """

    docs = []

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder, filename)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        docs.append({
            "title": filename,
            "text": text
        })

    return docs


def chunk_text(text: str) -> List[str]:
    """
    Split one text file into anime entries using
    <ANIME> ... </ANIME>.
    """

    pattern = r"<ANIME>(.*?)</ANIME>"

    matches = re.findall(
        pattern,
        text,
        flags=re.DOTALL
    )

    return [m.strip() for m in matches]


def build_chunk_records(docs: List[dict]) -> List[Chunk]:
    """
    Convert every anime into one Chunk object.
    """

    records = []

    for doc in docs:

        anime_list = chunk_text(doc["text"])

        for anime in anime_list:

            uid = re.search(r"UID:\s*(.*)", anime)
            title = re.search(r"Title:\s*(.*)", anime)
            rank = re.search(r"Rank:\s*(.*)", anime)
            genre = re.search(r"Genre:\s*(.*)", anime)

            metadata = {
                "uid": uid.group(1).strip() if uid else "",
                "title": title.group(1).strip() if title else "",
                "rank": rank.group(1).strip() if rank else "",
                "genre": genre.group(1).strip() if genre else "",
                "source": doc["title"]
            }

            records.append(
                Chunk(
                    chunk_id=metadata["uid"],
                    doc_title=metadata["title"],
                    text=anime,
                    metadata=metadata
                )
            )

    return records