import unittest
from types import SimpleNamespace
from unittest.mock import patch

import arxiv

from util.request import get_yesterday_arxiv_papers


def make_result(index):
    arxiv_id = f"2501.{index:05d}"
    return SimpleNamespace(
        title=f"Paper {index}",
        entry_id=f"https://arxiv.org/abs/{arxiv_id}v1",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}v1",
        summary=f"Abstract {index}",
        authors=[SimpleNamespace(name=f"Author {index}")],
        categories=["quant-ph"],
        primary_category="quant-ph",
        comment=None,
        get_short_id=lambda: arxiv_id,
    )


class ArxivRequestTest(unittest.TestCase):
    @patch("util.request.arxiv.Client")
    def test_uses_arxiv_package_and_preserves_output_format(self, mock_client_class):
        mock_client = mock_client_class.return_value
        mock_client.results.return_value = [make_result(0)]

        papers = get_yesterday_arxiv_papers("quant-ph", 5)

        search = mock_client.results.call_args.args[0]
        self.assertEqual(search.query, "cat:quant-ph")
        self.assertEqual(search.max_results, 5)
        self.assertEqual(search.sort_by, arxiv.SortCriterion.SubmittedDate)
        self.assertEqual(search.sort_order, arxiv.SortOrder.Descending)
        self.assertEqual(len(papers), 1)
        self.assertEqual(
            papers[0],
            {
                "title": "Paper 0",
                "arXiv_id": "2501.00000",
                "abstract": "Abstract 0",
                "authors": ["Author 0"],
                "categories": ["quant-ph"],
                "comments": "No comments available",
                "pdf_url": "https://arxiv.org/pdf/2501.00000v1",
                "abstract_url": "https://arxiv.org/abs/2501.00000v1",
            },
        )

    @patch("util.request.arxiv.Client")
    def test_truncates_results_to_max_results(self, mock_client_class):
        mock_client = mock_client_class.return_value
        mock_client.results.return_value = [make_result(index) for index in range(10)]

        papers = get_yesterday_arxiv_papers("quant-ph", 5)

        self.assertEqual(len(papers), 5)
        self.assertEqual(papers[-1]["arXiv_id"], "2501.00004")

    @patch("util.request.time.sleep")
    @patch("util.request.arxiv.Client")
    def test_transient_errors_retry_then_return_empty_list(
        self, mock_client_class, mock_sleep
    ):
        mock_client = mock_client_class.return_value
        mock_client.results.side_effect = RuntimeError("temporary network error")

        papers = get_yesterday_arxiv_papers("quant-ph", 5)

        self.assertEqual(papers, [])
        self.assertEqual(mock_client.results.call_count, 4)
        self.assertEqual(
            [call.args[0] for call in mock_sleep.call_args_list],
            [5, 15, 30],
        )

    @patch("util.request.time.sleep")
    @patch("util.request.arxiv.Client")
    def test_transient_error_can_recover(self, mock_client_class, mock_sleep):
        mock_client = mock_client_class.return_value
        mock_client.results.side_effect = [
            RuntimeError("temporary network error"),
            [make_result(0)],
        ]

        papers = get_yesterday_arxiv_papers("quant-ph", 5)

        self.assertEqual(len(papers), 1)
        self.assertEqual(mock_client.results.call_count, 2)
        self.assertEqual([call.args[0] for call in mock_sleep.call_args_list], [5])


if __name__ == "__main__":
    unittest.main()
