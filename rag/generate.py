"""
Generation: combines retrieved anime documents with Gemini to answer questions.

Modes:
- extractive: returns retrieved chunks directly
- llm: uses gemini-3.1-flash-lite with retrieved context
"""

import os
from typing import List, Tuple

from google import genai

from .ingest import Chunk


# -------------------------
# Extractive Mode
# -------------------------

def extractive_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    if not retrieved:
        return "No relevant anime were found."

    lines = [f"Retrieved anime for: {query}\n"]

    for chunk, score in retrieved:
        lines.append(
            f"Source: {chunk.doc_title} "
            f"(Similarity: {score:.3f})\n"
        )
        lines.append(chunk.text)
        lines.append("\n" + "=" * 70 + "\n")

    return "\n".join(lines)


# -------------------------
# Gemini Mode
# -------------------------

def llm_answer(
    query: str,
    retrieved: List[Tuple[Chunk, float]],
    history: List[dict] = None,
) -> str:

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "GEMINI_API_KEY was not found.\n\n"
            "Set it in your environment or .env file.\n\n"
            + extractive_answer(query, retrieved)
        )

    if not retrieved:
        return "I couldn't find any relevant anime."

    client = genai.Client(api_key=api_key)

    context = "\n\n".join(
        chunk.text
        for chunk, _ in retrieved
    )

    # Format conversation history if available
    history_context = ""
    if history:
        history_context = "Conversation History:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_context += f"- {role}: {msg['content']}\n"
        history_context += "\n"

    prompt = f"""
You are an anime recommendation assistant.

You MUST answer ONLY using the information inside the context.

Rules:
- Do not invent anime.
- Do not use outside knowledge.
- If the answer is not contained in the context, say:
  "I couldn't find that information in my anime database."
- When recommending anime, briefly explain why.
- Mention the anime title(s).

{history_context}Context
========
{context}

Question:
{query}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


# -------------------------
# Main Interface
# -------------------------

def generate_answer(
    query: str,
    retrieved: List[Tuple[Chunk, float]],
    mode: str = "extractive",
    history: List[dict] = None,
) -> str:

    if mode == "llm":
        return llm_answer(query, retrieved, history)

    return extractive_answer(query, retrieved)