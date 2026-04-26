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
            raise ValueError("GEMINI_API_KEY not found in environment variables.")

        # Initialize Gemini Client
        self.client = genai.Client(api_key=self.api_key)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.chroma_client.get_or_create_collection(
            name="steam_games",
            metadata={"hnsw:space": "cosine"}
        )

        self.records = self.load_records()
        # Run indexing in a background thread so the app starts immediately
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
                metadatas.append({"name": r.name, "app_id": r.app_id})

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

    def search(self, query: str) -> dict[str, Any]:
        candidates = self.retrieve_candidates(query)
        ranked_matches = self.rank_candidates(query, candidates)
        results = [record.to_result(score) for record, score in ranked_matches]

        return {
            "matches": results,
            "answer": self.generate_answer(query, ranked_matches),
            "meta": {
                "indexed_games": len(self.records),
                "retrieval_mode": "random-demo",
                "note": "Replace the scaffold in recommender.py with your own SQLite-backed retrieval, ranking, and LLM logic.",
            },
        }

    def retrieve_candidates(self, query: str) -> list[GameRecord]:
        """
        Retrieves candidates from ChromaDB based on vector similarity.
        """
        try:
            # Query ChromaDB (it will automatically embed the query)
            results = self.collection.query(
                query_texts=[query],
                n_results=DEFAULT_MATCH_COUNT * 2  # Get more so we can rank/filter
            )

            # Map back to GameRecord objects
            candidates = []
            seen_ids = set()
            
            # results['ids'][0] contains the app_ids of the matches
            for app_id in results['ids'][0]:
                if app_id in seen_ids:
                    continue
                # Find the record in self.records
                for record in self.records:
                    if record.app_id == app_id:
                        candidates.append(record)
                        seen_ids.add(app_id)
                        break
            
            return candidates
        except Exception as e:
            print(f"Retrieval error: {e}")
            # Fallback to random if something goes wrong
            if not self.records:
                return []
            return random.sample(self.records, min(DEFAULT_MATCH_COUNT, len(self.records)))

    def rank_candidates(
        self, query: str, candidates: list[GameRecord]
    ) -> list[tuple[GameRecord, float]]:
        """
        Ranks candidates by calculating their similarity to the query.
        """
        if not candidates:
            return []

        try:
            # Get embeddings for all candidates
            texts = []
            for r in candidates:
                tags = ", ".join(r.raw.get("tags", {}).keys()) if isinstance(r.raw.get("tags"), dict) else ""
                genres = ", ".join(r.raw.get("genres", []))
                text = f"Name: {r.name}\nGenres: {genres}\nTags: {tags}\nDescription: {r.short_description}"
                texts.append(text)

            # Let ChromaDB return the distances, we can just use those.
            # But wait, we didn't save distances in retrieve_candidates.
            # For simplicity, we can just return a dummy score or re-query to get distances.
            # Since retrieve_candidates actually already returns the best ones, we can just assign descending scores.
            ranked = []
            for i, record in enumerate(candidates):
                score = 1.0 - (i * 0.1)  # Dummy descending score since Chroma already ranked them
                ranked.append((record, float(score)))

            return ranked[:DEFAULT_MATCH_COUNT]
        except Exception as e:
            print(f"Ranking error: {e}")
            return [(r, 1.0 - (i / len(candidates))) for i, r in enumerate(candidates[:DEFAULT_MATCH_COUNT])]

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
        for record, score in matches[:3]:
            reviews = self._get_top_reviews(record.app_id)
            review_text = "\n".join([f"- {rev[:200]}..." for rev in reviews])
            item = (
                f"Game: {record.name}\n"
                f"Description: {record.short_description}\n"
                f"Genres: {', '.join(record.raw.get('genres', []))}\n"
                f"Player Reviews:\n{review_text}"
            )
            context_items.append(item)

        context_str = "\n\n".join(context_items)
        
        prompt = (
            f"You are a Steam game recommendation expert. A user is looking for games with this description: \"{query}\"\n\n"
            f"Based on the store metadata and player reviews, here are some top matches:\n\n"
            f"{context_str}\n\n"
            f"Please provide a helpful, engaging recommendation. Explain why these games match the user's interest. "
            f"Reference specific details from the descriptions or player feedback provided. Keep it concise (2-3 paragraphs)."
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
