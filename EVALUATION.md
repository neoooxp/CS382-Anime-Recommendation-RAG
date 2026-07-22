# Evaluation

Test the 10 plot-grounded queries below in your Streamlit chat interface (using **LLM mode**). Fill out the table with your actual results.

### 📊 Evaluation Table

| # | Query | Top Source Retrieved (Highest Similarity) | Similarity Score | Retrieval Correct? (Yes/No) | Answer Quality & Citation Notes (LLM Mode) |
|---|-------|-------------------------------------------|------------------|-----------------------------|-------------------------------------------|
| 1 | What is a popular action anime about human remnants fighting giants attacking a walled city? | City Hunter 2 | 0.464 | No | Bot response "I couldn't find that information in my anime database." and the other 2 were Giant Robo the Animation: Chikyuu ga Seishi Suru Hi, and City Hunter |
| 2 | Recommend a popular sci-fi anime about a mad scientist and time travel. | Summer Wars | 0.424 | No | Bot response "I couldn't find that information in my anime database." and the other 2 were Terra e... (TV), and Tsumiki no Ie |
| 3 | What anime involves two brothers trying to retrieve their bodies using alchemy and the Philosopher's Stone? | Fullmetal Alchemist | 0.543 | Yes | Bot response "I recommend both Fullmetal Alchemist and Fullmetal Alchemist: Brotherhood. Both anime follow the brothers Edward and Alphonse Elric as they search for the Philosopher's Stone in an attempt to restore their bodies after a failed alchemical experiment." and the other 2 was fullmetal alchemist brotherhood and special |
| 4 | Is there a sports anime centered around a high school volleyball team? | Haikyuu!! Second Season | 0.560 | Yes | Bot response "Yes, I recommend Haikyuu!! and Haikyuu!! Second Season. These anime center on the Karasuno High School volleyball team as the members work to overcome their differences, refine their skills, and become the best team in Japan." and the other 2 was Haikyuu and Ookiku Furikabutte: Natsu no Taikai-hen |
| 5 | What anime features a spaceship called the Bebop and a team of bounty hunters? | Cowboy Bebop: Tengoku no Tobira | 0.567 | Yes | Bot response"I recommend Cowboy Bebop and Cowboy Bebop: Tengoku no Tobira. These titles feature a ragtag crew of bounty hunters who travel and live aboard the spaceship Bebop while pursuing their targets across the solar system." The other 2 was cowboy bebop and Uchuu Kaizoku Captain Herlock |
| 6 | Recommend a supernatural action anime featuring samurai or demons. | Hoozuki no Reitetsu 2nd Season | 0.587 | Idk actually | Bot response "I recommend Shaman King and Hataraku Maou-sama!. Shaman King features supernatural action where characters command spirits, including a samurai ghost named Amidamaru. Hataraku Maou-sama! focuses on a demon lord who is stranded in modern-day Tokyo and must adapt to life as a human." The second and third were Shaman King and Hataraku Maou-sama!, and the bot answered with them instead |
| 7 | What romance anime involves a high school girl and boy who swap bodies? | Kimi to Boku. | 0.594 | idk actually | The bot just said couldn't find in database, Gekkan Shoujo Nozaki-kun, Yamato Nadeshiko Shichihenge♥, second and third btw |
| 8 | Tell me about a sports anime about a bullied boy who gets trained to become a professional boxer. | Ashita no Joe | 0.540 | Um idk | It just replied with one word "None" and heres the second and third, Hajime no Ippo: Boxer no Kobushi, Ashita no Joe 2 |
| 9 | What anime follows Izuku "Deku" Midoriya as he trains to become a superhero at a school for heroes? | Boku no Hero Academia | 0.725 | Yes | Bot response "I recommend Boku no Hero Academia, Boku no Hero Academia 2nd Season, and Boku no Hero Academia the Movie 1: Futari no Hero. These titles follow Izuku Midoriya, who inherits a unique power from his idol All Might and enrolls in UA High, a prestigious school for hero training, to learn what it means to be a hero while facing villains and competing against his talented classmates." The other 2 chunks were also my hero academia related |
| 10| Is there a cooking-themed anime about a student named Souma attending an elite culinary school? | Shokugeki no Souma | 0.714 | I think? | The bot's response "Yes, I recommend Shokugeki no Souma, Shokugeki no Souma: Ni no Sara, and Shokugeki no Souma: San no Sara - Tootsuki Ressha-hen. These anime follow Souma Yukihira as he attends the elite Tootsuki Culinary Academy, where he participates in high-stakes cooking showdowns known as "food wars" and faces various challenges to improve his culinary skills." and the other 2, Shokugeki no Souma: San no Sara - Tootsuki Ressha-hen, and Shokugeki no Souma: Ni no Sara  |

---

## 📝 Written Analysis

Based on the run results of the 10 evaluation queries, here is the performance evaluation of the upgraded RAG system:

### 🟢 What Worked
- **Specific Entities & Jargon**: Queries containing highly specific nouns, names, or terminology (e.g., `"volleyball"`, `"alchemy"`, `"Philosopher's Stone"`, `"Bebop"`, `"Deku Midoriya"`, `"Souma"`) retrieved the correct documents with high similarity scores (`0.540` to `0.725`). 
- **LLM Multi-Source Synthesis**: The LLM (`gemini-3.1-flash-lite`) did an excellent job of merging context across multiple related chunks (such as recommending both seasons of *Haikyuu!!* or *Boku no Hero Academia* and citing them correctly).
- **Active Context Reasoner**: In Query 6 (*supernatural action with samurai or demons*), the vector search returned *Hoozuki no Reitetsu* (a dark comedy about hell's bureaucracy) as the top similarity match. However, the LLM read the 2nd and 3rd chunks (*Shaman King* and *Hataraku Maou-sama!*), realized they matched the user's specific request for "samurai and demons" far better, and correctly recommended those instead.
- **Hallucination Prevention**: For failed retrievals, the LLM consistently fell back to the safety message (*"I couldn't find that information..."*), refusing to hallucinate outside of its context window.

### 🔴 What Didn't
- **Plot Premise Queries**: General queries describing the initial premise of an anime using common English vocabulary (e.g., `"giants attacking a walled city"`, `"swap bodies"`, or `"bullied boy boxer"`) failed to retrieve the correct main series (*Attack on Titan*, *Your Name*, and *Hajime no Ippo*).
- **Grounding Quirks**: For Query 8, the LLM simply replied with `"None"` rather than the full safety fallback string when it could not locate the bullied boxer backstory in the retrieved chunks.

### 🔍 Why (Root Causes)
1. **Semantic/Lexical Gap**: Synopses in the database do not always use standard layperson words. For example:
   - *Attack on Titan* uses `"Titans"` and `"Wall Maria"`, not `"giants"` and `"walled city"`.
   - *Your Name* (*Kimi no Na wa.*) describes the characters as `"living in each other's body"`, not using the phrase `"swap bodies"`.
2. **Sequel-Heavy Corpus**: The database frequently indexes sequels (e.g., *Shingeki no Kyojin Season 3 Part 2* or *Haikyuu!! Second Season*) instead of Season 1. Sequel synopses focus on specific sequel subplots (like retaking Wall Maria) rather than restating the show's initial premise, leading to poor matches for entry-level questions.
3. **Top-K Retrieval Constraints**: For *Hajime no Ippo*, the search retrieved *Ashita no Joe* and the *Hajime no Ippo: Boxer no Kobushi* special, but missed the main series description. Setting `top_k=3` cut off the main series entry before it could reach the LLM's context window.
