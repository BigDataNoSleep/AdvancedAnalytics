0. How to Run This Project (Evaluator Guide)

* **Prerequisites**: Ensure Python 3.10+ is installed. The codebase and all necessary files are attached and also available on GitHub.
* **API Key**: The system requires the Google Gemini API to function. For ease of testing, my personal `GEMINI_API_KEY` is already included directly within the code, meaning no external account setup is required on your end.
* **Setup**: Install the required dependencies. It is recommended to use `uv` for a fast and reliable setup:
    ```bash
    cd raglooker
    uv sync
    ```
    *Alternatively, using standard pip:*
    ```bash
    pip install flask ollama rank-bm25 chromadb sentence-transformers google-genai python-dotenv matplotlib scikit-learn numpy
    ```
* **Execution**: Launch the application using `uv`:
    ```bash
    uv run app.py
    ```
    *Alternatively, using standard python:*
    ```bash
    python app.py
    ```
* **First Boot**: Upon startup, a background daemon thread will automatically instantiate the ChromaDB vector database and BM25 index. Open your web browser and navigate to the local host address provided in the terminal (usually `http://127.0.0.1:5000`) to interact with the RAG interface. Note that the first search might take slightly longer as the local Cross-Encoder model initializes in memory.

⸻

1. Introduction

* **Brief description of the assignment**: The objective of this assignment is to design and develop an intelligent, Large Language Model (LLM)-based recommender system tailored for Steam games. It goes beyond simple keyword matching by understanding the semantic intent of complex natural language user queries.
* **Goal**: To build a robust pipeline that leverages Retrieval-Augmented Generation (RAG) to fetch highly relevant game candidates and dynamically generate personalized, conversational, and context-aware recommendations for users based on game metadata and community player reviews.
* **Short summary of your approach**: The implemented solution utilizes a highly advanced, multi-stage architecture. It begins with LLM-powered Self-Querying for query expansion and metadata extraction (acting as a guardrail). Candidate retrieval is performed via Hybrid Search combining Vector similarity (ChromaDB) and Keyword matching (BM25), merged using Reciprocal Rank Fusion (RRF). Retrieved candidates are re-ranked using a dedicated Cross-Encoder model to maximize precision. Finally, a generative LLM (Gemma 3 12B IT) constructs the final response, injecting rich context including price, tags, reception, and top community reviews to justify the recommendations.

⸻

2. Dataset Description

* **Source of the data**: Steam scraping, April 2026 snapshot.
* **Structure of the SQLite database (`steam_games_reviews_25.sqlite`)**:
    * **`games` table**: Contains rich metadata for games including `app_id` (Primary Key), `name`, `short_description`, `genres`, `tags`, `price`, `release_date`, `header_image`, platform compatibility (`windows`, `mac`, `linux`), and reception metrics (`positive` and `negative` review counts).
    * **`reviews` table**: Contains user-generated reviews, linking via `appid`. Fields include the review text, language, and a `weighted_vote_score` to determine review helpfulness.
* **Key characteristics**:
    * **Number of games**: The system is configured to load and index a maximum of 5,000 games to ensure optimal memory usage and indexing speed while providing a sufficiently diverse catalog.
    * **Review limits**: Reviews are dynamically fetched per game with a limit (top 3 reviews based on `weighted_vote_score`), filtered strictly for `language = 'english'` to maintain generation quality and prompt adherence.
* **Any preprocessing you performed**:
    * Tag normalization: Extracted the top 8 tags from nested dictionary/list structures to prevent context bloating.
    * Feature construction: Concatenated `Name`, `Genres`, `Tags`, and `Description` into a single, cohesive document string (with explicit prefixes) to enrich the semantic context for both vector embeddings and BM25 tokenization.

⸻

3. System Architecture

* **High-level overview of your pipeline**: The pipeline is designed for high precision and observability (via Server-Sent Events logging exposed to the frontend). It processes queries sequentially through query rewriting, hybrid retrieval, cross-encoder re-ranking, and response generation.
* **Components**:
    * **Data loading**: SQLite records are loaded into memory as `GameRecord` dataclasses. Vector indexing into ChromaDB runs asynchronously in a background daemon thread to prevent blocking Flask reloads.
    * **Prompt Cleaning & Self-Querying (First LLM Call)**: Before any retrieval occurs, the raw user prompt is intercepted by an initial, dedicated LLM call (`Gemma-3-12b-it`). This critical step "cleans" the prompt: it translates messy, conversational queries into an optimized, dense semantic search string (focusing on keywords, mechanics, and genres). Additionally, it acts as a self-querying guardrail by extracting hard metadata filters (e.g., converting "I want a free game" into a structured JSON payload like `{"filters": {"is_free": true}}`) and instantly rejecting completely off-topic queries (like recipe requests) by returning a `NOT_RELEVANT` flag. This ensures the vector database only receives high-quality search terms.
    * **Retrieval system**: A dual-engine approach. ChromaDB handles dense vector retrieval, while BM25 handles sparse exact-keyword retrieval.
    * **Ranking**: Results from both retrievers are merged via Reciprocal Rank Fusion (RRF), then passed to a Sentence Transformers Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) for pairwise relevance scoring.
    * **LLM generation**: Top-K matches and their associated top English reviews are compiled into an augmented prompt and fed back to the Gemma model for final answer generation.

⸻

4. Embedding Strategy

* **Model used**: ChromaDB's default local Sentence Transformer model (`all-MiniLM-L6-v2`).
    * *Note on Gemini Embeddings*: The initial design intended to use Google's `gemini-embedding-2` model for richer semantic understanding. However, during implementation, the strict Requests Per Minute (RPM) limits of the free Gemini API tier caused severe bottlenecking. Batch embedding 5,000 games took unacceptably long and frequently failed due to rate limiting. Consequently, the architecture was pivoted to use local, on-device embeddings via ChromaDB to ensure stable and rapid deployment.
* **What you embedded**: A constructed textual document combining four key metadata fields:
    * Game Name (`name`)
    * Genres (comma-separated string)
    * Tags (comma-separated string of the top extracted tags)
    * Game descriptions (`short_description`)
* **Preprocessing steps**: 
    * Checked and handled missing or malformed tag dictionaries to prevent type errors.
    * Formatted strings with explicit prefixes (e.g., "Name: ... \nGenres: ...") to artificially structure the text and guide the embedding model's semantic understanding.
* **Embedding storage**: 
    * **ChromaDB**: Used as a Persistent vector database (`.chroma_db` directory) configured with an HNSW index and `cosine` distance space for similarity measurements.
    * Batched upserts (20 documents per batch) with built-in retry mechanisms and database existence checks to ensure robust initialization.

⸻

5. Candidate Retrieval (`retrieve_candidates`)

* **How you match user query to games**: Utilizing a Hybrid Search methodology.
    * **Vector Search**: Queries ChromaDB for semantic similarity against the optimized query string, fetching the top 30 candidates.
    * **Keyword Search (BM25)**: Evaluates exact keyword matches using the BM25Okapi algorithm against a tokenized corpus of names, descriptions, and genres, fetching the top 30 candidates.
* **Similarity metric**: Cosine similarity for Vector Search; Probabilistic Term Frequency-Inverse Document Frequency (TF-IDF) logic for BM25.
* **Number of candidates retrieved**: 30 candidates per retrieval method.
* **Any filtering**: 
    * **Self-Querying Metadata Filtering**: The LLM analyzes the raw query for explicit filters. For instance, if the user asks for "free games", a hard filter (`{"price": 0.0}`) is injected dynamically into the ChromaDB `where` clause and applied manually as a list comprehension to the BM25 results list.
    * **Result Merging**: The two candidate sets are merged and re-scored using Reciprocal Rank Fusion (RRF) with a constant k = 60 to generate a unified list of the top 30 best candidates overall.

⸻

6. Candidate Ranking (`rank_candidates`)

* **How you refine the retrieved candidates**: The top 30 candidates from the RRF step are passed to a highly accurate Cross-Encoder for pairwise relevance prediction.
* **Use of**:
    * **LLM-based re-ranking / Cross-Encoder**: Utilized `cross-encoder/ms-marco-MiniLM-L-6-v2`. It takes the optimized query and the fully formatted game document (Name, Genres, Tags, Description) as pairs, outputting a highly calibrated relevance score for each pair.
    * **Heuristics**: A rule-based exclusion filter ensures the final output drops any game explicitly mentioned by the user in their raw query string, guaranteeing the system provides *new* recommendations rather than simply reiterating known games.
* **Final selection strategy**: Candidates are sorted descending by their predicted Cross-Encoder score, and the top 5 are passed to the context window for answer generation.

⸻

7. Answer Generation (`generate_answer`)

* **Prompt design**: 
    * **Structure of the prompt**: A highly constrained system prompt instructing the model to act as a Steam recommendation expert. It dictates that the model must respect the strict ranking order (Rank #1 down) and present a structured, engaging summary.
    * **Context included**: Rank, Relevance Score, Game Name, Price, Release Date, Top Tags, Reception Metrics (Positive vs Negative reviews), Short Description, and up to 3 helpful English player reviews per game.
* **Model used for generation**: Google Gemini API running `models/gemma-3-12b-it` (selected for its high quota allowance and strong instruction-following capabilities).
* **How you ensure**:
    * **Relevance**: Ensured by the rigorous Hybrid Search + Cross-Encoder re-ranking pipeline. Off-topic queries are blocked entirely before reaching generation via the self-querying guardrail.
    * **Diversity**: Answer diversity is naturally driven by the varied games returned via Hybrid Retrieval, which prevents over-fitting to just semantic or just keyword matches.
    * **Readability**: The prompt explicitly enforces the use of provided metadata (like price and tags) and instructs the model to explicitly reference player feedback from the reviews, ensuring an objective, conversational tone.

⸻

8. Use of Reviews (Optional but Strong Section)

* **Whether and how reviews were incorporated**: Highly integral to the final generation phase. The system executes a real-time SQLite query (`SELECT review FROM reviews WHERE appid = ? AND language = 'english' ORDER BY weighted_vote_score DESC LIMIT 3`) for every top candidate entering the LLM generation stage.
* **Selection Logic (What makes them "relevant"?)**:
    * **Community Consensus**: Rather than performing a second semantic search for specific review snippets (which would be computationally expensive), the system relies on **Steam's community helpfulness ranking**. 
    * **`weighted_vote_score`**: Reviews are sorted strictly by this score. This metric represents a "weighted" score of how many users found the review helpful, filtered for the English language.
    * **Quality Guardrail**: By selecting reviews with the highest community agreement, the system ensures the LLM receives the most **trusted, representative, and informative** feedback. This filters out "joke" reviews or low-effort comments, providing the model with substantive player experiences to cite.
* **Impact on recommendation quality**: The prompt forces the LLM to explicitly justify *why* the game fits the query based on these high-quality reviews. This allows the model to catch nuances—like "steep learning curve" or "best with friends"—that are often missing from developer-written short descriptions, resulting in a significantly more grounded and honest recommendation.

⸻

9. Experiments and Evaluation

* **How you evaluated your system**:
    * **Manual testing**: Conducted structured qualitative tests covering diverse scenarios (genre-specific, obscure descriptors, free-to-play constraints, and completely unrelated/off-topic inputs).
    * **Observability validation**: Utilized the frontend's real-time Server-Sent Events (SSE) logs to trace the self-querying expansions, RRF scoring, and filter application accurately, confirming the pipeline logic was firing correctly.
* **Example queries and outputs**:
    * *Query*: "I want a free multiplayer shooter like CS:GO" -> The self-querying accurately extracted `{"is_free": true}`. The hybrid system retrieved games like Team Fortress 2, and the LLM explicitly quoted user reviews about the gunplay.
    * *Query*: "How do I bake a cake?" -> The guardrail accurately predicted `NOT_RELEVANT`, bypassing the RAG pipeline entirely and returning a graceful off-topic denial.
* **Observations**:
    * **Strengths**: Incredibly high precision due to the Cross-Encoder. The LLM's inclusion of review sentiment drastically reduces "hallucinated" hype, yielding grounded, realistic recommendations.
    * **Weaknesses**: Latency. The sequential nature of Self-Querying (LLM) -> BM25 + Vector Search -> Cross-Encoder -> LLM Generation introduces noticeable processing time for the user.

⸻

10. Challenges and Limitations

* **Technical challenges**:
    * **Performance Latency**: Executing two remote LLM calls (Self-Querying and Generation) plus a local Cross-Encoder dynamically per query creates a significant response bottleneck.
    * **ChromaDB Metadata Queries**: Early indexing iterations without explicit price float values broke the `where` clause filtering, requiring a fallback try-except block to gracefully degrade to standard vector search without metadata if the schema was mismatched.
    * **JSON Adherence**: The Self-Querying LLM occasionally prefixed its output with markdown formatting (` ```json `), necessitating custom parsing logic (`text[7:-3].strip()`) to prevent pipeline JSON decoder crashes.
* **Limitations**:
    * **Dataset bias**: Maxing out at 5,000 games limits the long-tail discovery capability of truly obscure indie titles.
    * **Embedding quality**: Relying on local `all-MiniLM-L6-v2` embeddings limits context depth compared to larger enterprise embeddings (e.g., OpenAI `text-embedding-3-large`).

⸻

11. Improvements and Future Work

* **Ideas for improvement**:
    * **Asynchronous Fetching**: Fetch the SQLite top reviews for the top candidates asynchronously using Python's `asyncio` or ThreadPoolExecutor to reduce the latency of the answer generation prep phase.
    * **Better models**: Upgrade the local vector embeddings to a larger parameter model to capture deeper semantic relationships.
    * **Personalisation**: Inject the user's Steam ID to pull their currently owned games, dropping them from the recommendation pool while weighting genres they frequently play higher in the Self-Querying expansion step.
    * **Caching Layer**: Introduce a Redis caching layer for the `retrieve_candidates` results for identical optimized search queries to drastically cut down Cross-Encoder computation times.

⸻

12. Conclusion

* **Short recap of what you built**: We engineered an advanced, multi-stage RAG-based Steam game recommender utilizing LLM Self-Querying guardrails, Hybrid Retrieval (BM25 + Vector), Reciprocal Rank Fusion, Cross-Encoder re-ranking, and dynamic Steam review injection.
* **Key insights**: Combining semantic vector search with exact keyword matching (BM25) prevents the loss of specific game names or niche mechanics. Injecting highly rated community reviews into the generative context grounds the LLM, preventing marketing fluff and resulting in highly trustworthy recommendations.
* **Final reflection on LLM-based recommenders**: LLM-based systems excel at parsing the nuance of natural language ("games that make me feel like a space pirate"), which traditional collaborative filtering struggles with. However, the architectural complexity and latency trade-offs require careful pipeline engineering to remain viable for real-time, user-facing applications.

⸻

13. Appendix (Optional)

* **Example prompts**:
    * *Self-Querying Prompt*: "You are a search query optimization expert... rewrite and expand this query to be optimal for vector similarity search... output ONLY valid JSON... If the query is completely unrelated to video games... output ONLY the word 'NOT_RELEVANT'."
    * *Answer Generation Prompt*: "You are a Steam game recommendation expert... respect ranking order... explicitly reference player feedback from the provided reviews... Be honest about drawbacks mentioned in the reviews or reception scores."
* **Visualization Tools**: A dedicated script (`visualize_embeddings.py`) was developed to perform dimensionality reduction (PCA) on the vector embeddings, generating a 2D scatter plot (`embeddings_visualization.png`) of the game catalog to verify semantic clustering.
* **Data Inspection**: Included an `inspect_db.py` utility to monitor the ChromaDB collection size, disk usage, and metadata integrity during the development process.
* **Architecture Note**: Implemented SSE (Server-Sent Events) to stream intermediate logs (query optimization, filter extraction, retrieval counts, and re-ranking) to the frontend application, providing vital system transparency to the end user.