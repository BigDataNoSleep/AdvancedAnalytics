from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from google import genai
from google.genai import types

import json
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from steam_sqlite import load_games_from_sqlite

# Load environment variables from .env file (searching up to find it)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))
CHROMA_PATH = BASE_DIR / ".chroma_db"
MAX_GAMES = 5000
DEFAULT_MATCH_COUNT = 5
EMBEDDING_MODEL = "models/gemini-embedding-2"
LLM_MODEL = "models/gemma-3-12b-it"  # High quota model as per user request




def create_search_engine() -> "GameSearchEngine":
    return GameSearchEngine(DB_PATH)


@dataclass
class GameRecord:
    app_id: str
    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return self.raw.get("name", "Unknown title")

    @property
    def short_description(self) -> str:
        return self.raw.get("short_description", "")

    def to_result(self, score: float) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "name": self.name,
            "score": round(score, 4),
            "short_description": self.short_description,
            "genres": self.raw.get("genres", []),
            "tags": self._normalize_tags(self.raw.get("tags")),
            "price": self.raw.get("price"),
            "release_date": self.raw.get("release_date"),
            "header_image": self.raw.get("header_image"),
            "store_page": f"https://store.steampowered.com/app/{self.app_id}",
            "platforms": {
                "windows": bool(self.raw.get("windows")),
                "mac": bool(self.raw.get("mac")),
                "linux": bool(self.raw.get("linux")),
            },
        }

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if isinstance(tags, dict):
            return list(tags.keys())[:8]
        if isinstance(tags, list):
            return tags[:8]
        return []


class GameSearchEngine:
    """
    This implementation is intentionally crude:
    - it loads a subset of games
    - it ignores the query for ranking
    - it returns random games as "matches"
    - it writes a simple canned answer instead of calling an LLM
    - it doesnt use any review information

    Suggested improvements:
    1. Replace `retrieve_candidates()` with keyword search, BM25, embeddings, or vector search.
    2. Replace `rank_candidates()` with a real ranking function.
    3. Replace `generate_answer()` with an LLM prompt over retrieved context.
    4. Try to make the LLM extract structured info (release year, genre, price) and narrow the matches using that.

    Keep the public `search()` return shape stable so the Flask app and frontend keep working.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            self.api_key = "AIzaSyC1OEagLcMnJLF0odjL54uUwRyJk5dHbjY"
            #raise ValueError("GEMINI_API_KEY not found in environment variables.")

        # Initialize Gemini Client
        self.client = genai.Client(api_key=self.api_key)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.chroma_client.get_or_create_collection(
            name="steam_games",
            metadata={"hnsw:space": "cosine"}
        )

        self.records = self.load_records()
        
        print("Initializing BM25 Index...")
        tokenized_corpus = [
            f"{r.name} {r.short_description} {' '.join(r.raw.get('genres', []))}".lower().split(" ")
            for r in self.records
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print("Initializing Cross-Encoder...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Run indexing in a background thread if not already started (prevents issues on Flask reloads)
        if not hasattr(GameSearchEngine, "_indexing_started"):
            GameSearchEngine._indexing_started = True
            threading.Thread(target=self.index_games, daemon=True).start()

    def load_records(self) -> list[GameRecord]:
        records: list[GameRecord] = []

        for app_id, raw in load_games_from_sqlite(self.db_path, MAX_GAMES):
            records.append(GameRecord(app_id=app_id, raw=raw))

        return records

    def index_games(self) -> None:
        """
        Indexes the games into ChromaDB if the collection is empty.
        """
        if self.collection.count() >= len(self.records):
            print(f"Collection already contains {self.collection.count()} games. Skipping indexing.")
            return

        print(f"Indexing {len(self.records)} games into ChromaDB...")
        
        batch_size = 20
        i = 0
        while i < len(self.records):
            batch = self.records[i:i + batch_size]
            ids = [r.app_id for r in batch]
            
            # Skip if the first ID in the batch is already in the collection
            try:
                res = self.collection.get(ids=[ids[0]])
                if res and res['ids']:
                    i += batch_size
                    continue
            except Exception:
                pass

            print(f"Indexing batch starting at index {i}...")
            texts = []
            metadatas = []
            for r in batch:
                tags = ", ".join(r.raw.get("tags", {}).keys()) if isinstance(r.raw.get("tags"), dict) else ""
                genres = ", ".join(r.raw.get("genres", []))
                text = f"Name: {r.name}\nGenres: {genres}\nTags: {tags}\nDescription: {r.short_description}"
                texts.append(text)
                metadatas.append({"name": r.name, "app_id": r.app_id, "price": float(r.raw.get("price", 0.0))})

            # Upsert into ChromaDB (Chroma will automatically embed the texts using its local model)
            try:
                self.collection.upsert(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
                print(f"Indexed {i + len(batch)} / {len(self.records)} games...")
                i += batch_size
            except Exception as e:
                print(f"Error indexing batch starting at index {i}: {e}")
                time.sleep(2)

        print("Indexing complete.")

    def rewrite_query(self, query: str) -> dict[str, Any]:
        """
        Uses the LLM to rewrite and expand the user's query into an optimal search query for the vector database.
        Also acts as a relevance guardrail and extracts hard metadata filters (Self-Querying).
        """
        prompt = (
            f"You are a search query guardrail and expansion expert for a Steam game recommendation engine.\n"
            f"A user entered the following search query: \"{query}\"\n\n"
            f"Please rewrite and expand this query to be optimal for vector similarity search against a database of game descriptions and tags.\n"
            f"Include relevant genres, sub-genres, gameplay mechanics, and thematic keywords that relate to the original query.\n"
            f"If the user mentions a specific game (like 'Theatre of War'), include the genres and themes of that game.\n"
            f"RULES:\n"
            f"1. If the query is about video games, Steam, or looking for game recommendations, rewrite and expand it to be optimal for vector similarity search (keywords, genres, mechanics, themes).\n"
            f"2. If the query is unrelated to video games (e.g., cooking recipes, medical advice, general chat, homework, ...), you can be strict(!), output ONLY the word 'NOT_RELEVANT'.\n"
            f"3. You must respond with ONLY valid JSON in the following format:\n"
            f"{{\n"
            f"  \"search_string\": \"keywords, genres, themes...\",\n"
            f"  \"filters\": {{\"is_free\": true_or_false}}\n"
            f"}}\n"
            f"Set 'is_free' to true ONLY if the user explicitly asks for a free game.\n"
            f"Do not write any markdown code blocks, just output the raw JSON."
        )
        try:
            response = self.client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt
            )
            text = response.text.strip()
            if "NOT_RELEVANT" in text.upper() and not text.startswith("{"):
                return {"search_string": "NOT_RELEVANT", "filters": {}}
            
            # Remove any markdown formatting if the LLM ignored instructions
            if text.startswith("```json"):
                text = text[7:-3].strip()
            elif text.startswith("```"):
                text = text[3:-3].strip()
                
            parsed = json.loads(text)
            print(f"Original Query: {query}")
            print(f"Self-Querying Result: {parsed}")
            return parsed
        except Exception as e:
            print(f"Query expansion/parsing error: {e}")
            return {"search_string": query, "filters": {}}

    def search_stream(self, raw_query: str):
        """
        Generator that yields Server-Sent Events for real-time pipeline logging.
        """
        import json as _json
        
        def log_event(msg: str) -> str:
            return f"event: log\ndata: {_json.dumps({'log': msg})}\n\n"
        
        # 1. Expand the query using the LLM (and check relevance / extract filters)
        yield log_event(f"📝 Original query: \"{raw_query}\"")
        query_data = self.rewrite_query(raw_query)
        optimized_query = query_data.get("search_string", raw_query)
        filters = query_data.get("filters", {})
        yield log_event(f"🔍 Optimized query: \"{optimized_query}\"")
        if filters:
            yield log_event(f"🏷️ Metadata filters: {_json.dumps(filters)}")
        
        if optimized_query == "NOT_RELEVANT":
            yield log_event("🚫 Query classified as off-topic. Aborting.")
            result = {
                "matches": [],
                "answer": "I'm sorry, but I can only assist with Steam game recommendations. Your query doesn't appear to be related to video games!",
                "meta": {
                    "indexed_games": len(self.records),
                    "retrieval_mode": "Guardrail Triggered",
                    "note": "Query was classified as off-topic.",
                },
            }
            yield f"event: result\ndata: {_json.dumps(result)}\n\n"
            return

        # 2. Retrieve candidates using Hybrid Search
        yield log_event("⚡ Running Hybrid Search (BM25 + ChromaDB Vector)...")
        candidates = self.retrieve_candidates(optimized_query, filters)
        yield log_event(f"📊 Retrieved {len(candidates)} candidates via Reciprocal Rank Fusion")
        
        # 3. Cross-Encoder Re-ranking
        yield log_event(f"🧠 Re-ranking {len(candidates)} candidates with Cross-Encoder...")
        ranked_matches = self.rank_candidates(optimized_query, candidates)
        if ranked_matches:
            yield log_event(f"🏆 Top match: \"{ranked_matches[0][0].name}\" (score: {ranked_matches[0][1]:.4f})")
        
        # 4. Filter out the game the user already mentioned (they want NEW games)
        filtered_matches = []
        for record, score in ranked_matches:
            if record.name.lower() in raw_query.lower():
                yield log_event(f"🗑️ Filtered out \"{record.name}\" (mentioned in query)")
                continue
            filtered_matches.append((record, score))
            
        results = [record.to_result(score) for record, score in filtered_matches[:DEFAULT_MATCH_COUNT]]
        yield log_event(f"✅ Returning top {len(results)} results")
        
        # 5. Generate LLM answer
        yield log_event(f"💬 Generating answer with {LLM_MODEL}...")
        answer = self.generate_answer(raw_query, filtered_matches[:DEFAULT_MATCH_COUNT])
        yield log_event("✅ Done!")

        result = {
            "matches": results,
            "answer": answer,
            "meta": {
                "indexed_games": len(self.records),
                "retrieval_mode": "Hybrid Search (BM25 + Vector) + Cross-Encoder Re-ranking",
                "note": "RAG System with Self-Querying, Reciprocal Rank Fusion, and Gemma 3 LLM",
            },
        }
        yield f"event: result\ndata: {_json.dumps(result)}\n\n"

    def retrieve_candidates(self, query: str, filters: dict[str, Any]) -> list[GameRecord]:
        """
        Retrieves candidates using Hybrid Search (BM25 + ChromaDB) and applies metadata filters.
        """
        try:
            n_retrieve = 30
            
            # 1. Apply Metadata Filters for ChromaDB
            where_clause = None
            if filters.get("is_free"):
                where_clause = {"price": 0.0}

            # 2. Vector Search (ChromaDB)
            try:
                chroma_results = self.collection.query(
                    query_texts=[query],
                    n_results=n_retrieve,
                    where=where_clause
                )
            except Exception:
                # Fallback: if where clause fails (e.g. old index without price metadata), retry without it
                print("ChromaDB where clause failed, retrying without metadata filter...")
                chroma_results = self.collection.query(
                    query_texts=[query],
                    n_results=n_retrieve
                )
            chroma_ids = chroma_results['ids'][0] if chroma_results['ids'] else []

            # 3. Keyword Search (BM25)
            tokenized_query = query.lower().split(" ")
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Sort records by BM25 score
            bm25_ranked = sorted(zip(self.records, bm25_scores), key=lambda x: x[1], reverse=True)
            
            # Apply hard filters to BM25 results
            if filters.get("is_free"):
                bm25_ranked = [(r, s) for r, s in bm25_ranked if r.raw.get("price", 1.0) == 0.0]
                
            bm25_ids = [r.app_id for r, _ in bm25_ranked[:n_retrieve]]

            # 4. Combine with Reciprocal Rank Fusion (RRF)
            k = 60
            rrf_scores: dict[str, float] = {}
            
            for rank, app_id in enumerate(chroma_ids):
                rrf_scores[app_id] = rrf_scores.get(app_id, 0.0) + 1.0 / (k + rank + 1)
                
            for rank, app_id in enumerate(bm25_ids):
                rrf_scores[app_id] = rrf_scores.get(app_id, 0.0) + 1.0 / (k + rank + 1)

            # Sort by RRF score
            sorted_app_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

            # Map back to GameRecord objects
            candidates = []
            for app_id in sorted_app_ids[:n_retrieve]:
                for record in self.records:
                    if record.app_id == app_id:
                        candidates.append(record)
                        break
            
            return candidates
        except Exception as e:
            print(f"Retrieval error: {e}")
            if not self.records:
                return []
            return random.sample(self.records, min(30, len(self.records)))

    def rank_candidates(
        self, query: str, candidates: list[GameRecord]
    ) -> list[tuple[GameRecord, float]]:
        """
        Ranks candidates using the Cross-Encoder model.
        """
        if not candidates:
            return []

        try:
            # Prepare pairs for the Cross-Encoder
            pairs = []
            for r in candidates:
                tags = ", ".join(r.raw.get("tags", {}).keys()) if isinstance(r.raw.get("tags"), dict) else ""
                genres = ", ".join(r.raw.get("genres", []))
                doc = f"Name: {r.name}\nGenres: {genres}\nTags: {tags}\nDescription: {r.short_description}"
                pairs.append((query, doc))
            
            # Predict scores
            scores = self.cross_encoder.predict(pairs)
            
            # Combine and sort (cast numpy float32 to Python float for JSON serialization)
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [(record, float(score)) for record, score in ranked]
        except Exception as e:
            print(f"Ranking error: {e}")
            return [(r, 1.0 - (i / len(candidates))) for i, r in enumerate(candidates)]

    def _get_top_reviews(self, app_id: str, limit: int = 3) -> list[str]:
        """
        Fetches the top reviews for a given app_id from the SQLite database.
        """
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                query = "SELECT review FROM reviews WHERE appid = ? AND language = 'english' ORDER BY weighted_vote_score DESC LIMIT ?"
                rows = conn.execute(query, (app_id, limit)).fetchall()
                return [row["review"] for row in rows if row["review"]]
        except Exception:
            return []

    def generate_answer(self, query: str, matches: list[tuple[GameRecord, float]]) -> str:
        """
        Generates a recommendation answer using Gemini (Gemma 3).
        """
        if not matches:
            return "I couldn't find any games matching your description."

        # Construct context for the LLM
        context_items = []
        for i, (record, score) in enumerate(matches):
            reviews = self._get_top_reviews(record.app_id)
            review_text = "\n".join([f"- {rev[:200]}..." for rev in reviews]) if reviews else "No English reviews available."
            
            tags_dict = record.raw.get("tags", {})
            tags = ", ".join(list(tags_dict.keys())[:5]) if isinstance(tags_dict, dict) else ""
            price = f"${record.raw.get('price')}" if record.raw.get("price") else "Free/Unknown"
            positive = record.raw.get("positive", 0)
            negative = record.raw.get("negative", 0)
            
            item = (
                f"Rank #{i+1} (Relevance Score: {score:.2f})\n"
                f"Game: {record.name}\n"
                f"Price: {price} | Released: {record.raw.get('release_date', 'Unknown')}\n"
                f"Top Tags: {tags}\n"
                f"Reception: {positive} Positive vs {negative} Negative reviews\n"
                f"Description: {record.short_description}\n"
                f"Player Reviews:\n{review_text}"
            )
            context_items.append(item)

        context_str = "\n\n".join(context_items)
        
        prompt = (
            f"You are a Steam game recommendation expert. A user is looking for games matching this query: \"{query}\"\n\n"
            f"I have retrieved the top {len(matches)} matches from the database, ordered strictly by relevance (Rank #1 is the best match).\n\n"
            f"--- MATCHES ---\n{context_str}\n-----------------\n\n"
            f"Please write a helpful, engaging recommendation for the user. Follow these rules strictly:\n"
            f"1. Respect the ranking order. Always present and discuss the games in order from Rank #1 down.\n"
            f"2. Use the provided metadata (price, tags, reception) to give a richer description.\n"
            f"3. Explicitly reference player feedback from the provided reviews to justify why these games fit their request.\n"
            f"4. Be honest about drawbacks mentioned in the reviews or reception scores (e.g., bad reviews, pay-to-win).\n"
            f"5. Keep your response well-structured and concise."
        )

        try:
            response = self.client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Generation error: {e}")
            names = ", ".join(record.name for record, _ in matches[:3])
            return f"I recommend checking out {names}. They seem to match your interest in {query}."
