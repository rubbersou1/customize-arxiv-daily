import json
import os
import sqlite3
from datetime import datetime, timezone

from loguru import logger


class PaperHistory:
    def __init__(self, db_path="data/papers.db"):
        self.db_path = db_path
        self.enabled = True
        try:
            db_dir = os.path.dirname(os.path.abspath(db_path))
            os.makedirs(db_dir, exist_ok=True)
            self._init_db()
        except sqlite3.Error as error:
            self.enabled = False
            logger.warning(f"SQLite history disabled: {error}")
        except OSError as error:
            self.enabled = False
            logger.warning(f"SQLite history disabled: {error}")

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_papers (
                    arxiv_id TEXT PRIMARY KEY,
                    title TEXT,
                    score REAL,
                    processed_at TEXT NOT NULL,
                    metadata_json TEXT
                )
                """
            )

    def has_seen(self, arxiv_id):
        if not self.enabled:
            return False
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT 1 FROM processed_papers WHERE arxiv_id = ?",
                    (arxiv_id,),
                ).fetchone()
            return row is not None
        except sqlite3.Error as error:
            logger.warning(f"Failed to read SQLite history for {arxiv_id}: {error}")
            return False

    def mark_processed(self, paper):
        if not self.enabled:
            return
        arxiv_id = paper.get("arXiv_id")
        if not arxiv_id:
            return
        metadata = {
            "authors": paper.get("authors", []),
            "categories": paper.get("categories", []),
            "arxiv_url": paper.get("arxiv_url", ""),
            "pdf_url": paper.get("pdf_url", ""),
            "local_pdf_path": paper.get("local_pdf_path", ""),
        }
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO processed_papers (
                        arxiv_id,
                        title,
                        score,
                        processed_at,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        title = excluded.title,
                        score = excluded.score,
                        processed_at = excluded.processed_at,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        arxiv_id,
                        paper.get("title", ""),
                        paper.get("relevance_score"),
                        datetime.now(timezone.utc).isoformat(),
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )
        except sqlite3.Error as error:
            logger.warning(f"Failed to write SQLite history for {arxiv_id}: {error}")

    def filter_unseen(self, papers):
        if not self.enabled:
            return papers
        return [paper for paper in papers if not self.has_seen(paper["arXiv_id"])]
