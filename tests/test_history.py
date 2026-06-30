import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from util.history import PaperHistory


class PaperHistoryTest(unittest.TestCase):
    def test_mark_processed_and_has_seen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "papers.db"
            history = PaperHistory(str(db_path))

            self.assertFalse(history.has_seen("2501.00001"))
            history.mark_processed(
                {
                    "arXiv_id": "2501.00001",
                    "title": "Quantum Sensing With Neutral Atoms",
                    "relevance_score": 8.5,
                    "authors": ["A. Researcher"],
                    "categories": ["quant-ph"],
                    "arxiv_url": "https://arxiv.org/abs/2501.00001",
                    "pdf_url": "https://arxiv.org/pdf/2501.00001",
                    "local_pdf_path": "data/pdfs/2026-06-30/2501.00001__paper.pdf",
                }
            )

            self.assertTrue(history.has_seen("2501.00001"))

            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    """
                    SELECT title, score, metadata_json
                    FROM processed_papers
                    WHERE arxiv_id = ?
                    """,
                    ("2501.00001",),
                ).fetchone()

            self.assertEqual(row[0], "Quantum Sensing With Neutral Atoms")
            self.assertEqual(row[1], 8.5)
            metadata = json.loads(row[2])
            self.assertEqual(metadata["categories"], ["quant-ph"])
            self.assertEqual(
                metadata["local_pdf_path"],
                "data/pdfs/2026-06-30/2501.00001__paper.pdf",
            )

    def test_mark_processed_updates_existing_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "papers.db"
            history = PaperHistory(str(db_path))

            history.mark_processed(
                {
                    "arXiv_id": "2501.00002",
                    "title": "Old Title",
                    "relevance_score": 7.0,
                }
            )
            history.mark_processed(
                {
                    "arXiv_id": "2501.00002",
                    "title": "New Title",
                    "relevance_score": 9.0,
                }
            )

            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT title, score FROM processed_papers WHERE arxiv_id = ?",
                    ("2501.00002",),
                ).fetchone()

            self.assertEqual(row, ("New Title", 9.0))

    def test_filter_unseen(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = PaperHistory(str(Path(tmpdir) / "papers.db"))
            history.mark_processed({"arXiv_id": "2501.00003", "title": "Seen"})

            papers = [
                {"arXiv_id": "2501.00003", "title": "Seen"},
                {"arXiv_id": "2501.00004", "title": "Unseen"},
            ]

            self.assertEqual(
                history.filter_unseen(papers),
                [{"arXiv_id": "2501.00004", "title": "Unseen"}],
            )


if __name__ == "__main__":
    unittest.main()
