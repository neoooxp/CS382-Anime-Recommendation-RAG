# Evaluation

Run `python scripts/run_eval.py` (set `EVAL_MODE=llm` with `GEMINI_API_KEY` set to
evaluate the full generation step; defaults to `extractive` so it runs with no
API key). Paste the actual output below and fill in the assessment column —
don't guess at results you haven't run.

| # | Query | Top source retrieved | Retrieval correct? | Answer quality notes |
|---|-------|----------------------|---------------------|------------------------|
| 1 | What's a good 2013 action anime about giants attacking a walled city? | *(fill in)* | | |
| 2 | Recommend a highly popular anime movie from the 1980s. | | | |
| 3 | What anime from 2006 involves alchemy and two brothers? | | | |
| 4 | Is there a sports anime about volleyball? | | | |
| 5 | What anime came out in 1998 involving bounty hunters in space? | | | |
| 6 | Recommend something from 2019 with a supernatural theme. | | | |
| 7 | What's the most popular romance anime in this collection? | | | |
| 8 | Tell me about an anime involving a boxing tournament from the 1990s. | | | |
| 9 | What anime is set in a school for superheroes? | | | |
| 10 | Is there an anime about competitive cooking? | | | |

## What to write once you have real results

- **What worked**: which query types retrieved cleanly (e.g. specific plot
  details vs. vague "recommend something" queries)?
- **What didn't**: any queries where a wrong-year entry got pulled in, or where
  the corpus genuinely doesn't contain an answer (e.g. genres/years not covered
  by the top-20-per-year cutoff) — check that `generate_answer` said so instead
  of hallucinating.
- **Why**: is it a retrieval problem (embedding model not distinguishing
  similar synopses), a chunking problem (an entry split awkwardly), or a
  coverage problem (your corpus caps at 20 anime/year, so niche titles are
  never even indexed)?
