import unittest
from unittest.mock import patch

from util.request import get_yesterday_arxiv_papers


class Response:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


def build_arxiv_html(count):
    entries = []
    for index in range(count):
        arxiv_id = f"2501.{index:05d}"
        entries.append(
            f"""
            <dt>
              <a title="Abstract" href="/abs/{arxiv_id}">abs</a>
              <a title="Download PDF" href="/pdf/{arxiv_id}">pdf</a>
            </dt>
            <dd>
              <div class="list-title">Title: Paper {index}</div>
              <div class="list-authors">Authors:
                <a>Author {index}</a>
              </div>
              <div class="list-subjects">
                Subjects: Quantum Physics (quant-ph)
              </div>
              <p class="mathjax">Abstract {index}</p>
            </dd>
            """
        )
    return f'<html><body><dl id="articles">{"".join(entries)}</dl></body></html>'


class ArxivRequestTest(unittest.TestCase):
    @patch("util.request.requests.get")
    def test_max_entries_5_requests_show_25_and_truncates(self, mock_get):
        mock_get.return_value = Response(build_arxiv_html(25))

        papers = get_yesterday_arxiv_papers("quant-ph", 5)

        self.assertIn("show=25", mock_get.call_args.args[0])
        self.assertEqual(len(papers), 5)
        self.assertEqual(papers[0]["arXiv_id"], "2501.00000")
        self.assertEqual(papers[0]["authors"], ["Author 0"])
        self.assertEqual(papers[0]["categories"], ["quant-ph"])

    @patch("util.request.requests.get")
    def test_max_entries_100_requests_show_100(self, mock_get):
        mock_get.return_value = Response(build_arxiv_html(100))

        papers = get_yesterday_arxiv_papers("quant-ph", 100)

        self.assertIn("show=100", mock_get.call_args.args[0])
        self.assertEqual(len(papers), 100)
        self.assertEqual(papers[-1]["arXiv_id"], "2501.00099")

    @patch("util.request.time.sleep")
    @patch("util.request.requests.get")
    def test_http_429_retries_then_returns_empty_list(self, mock_get, mock_sleep):
        mock_get.return_value = Response("Rate exceeded.", status_code=429)

        papers = get_yesterday_arxiv_papers("quant-ph", 100)

        self.assertEqual(papers, [])
        self.assertEqual(mock_get.call_count, 4)
        self.assertEqual(
            [call.args[0] for call in mock_sleep.call_args_list],
            [30, 60, 120],
        )

    @patch("util.request.time.sleep")
    @patch("util.request.requests.get")
    def test_http_429_retry_can_recover(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            Response("Rate exceeded.", status_code=429),
            Response(build_arxiv_html(1), status_code=200),
        ]

        papers = get_yesterday_arxiv_papers("quant-ph", 100)

        self.assertEqual(len(papers), 1)
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual([call.args[0] for call in mock_sleep.call_args_list], [30])


if __name__ == "__main__":
    unittest.main()
