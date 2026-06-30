"""
Use the official arxiv package to get recent arXiv papers.
"""

import time

import arxiv
from loguru import logger


ARXIV_TRANSIENT_BACKOFF_SECONDS = (5, 15, 30)


def _build_search(category: str, max_results: int):
    return arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )


def _get_short_id(result):
    if hasattr(result, "get_short_id"):
        return result.get_short_id()
    return result.entry_id.rstrip("/").split("/")[-1].split("v")[0]


def _normalize_result(result):
    arxiv_id = _get_short_id(result)
    abstract_url = result.entry_id
    pdf_url = result.pdf_url or f"https://arxiv.org/pdf/{arxiv_id}"
    authors = [author.name for author in getattr(result, "authors", [])]
    categories = list(getattr(result, "categories", []) or [])
    primary_category = getattr(result, "primary_category", None)
    if primary_category and primary_category not in categories:
        categories.insert(0, primary_category)

    return {
        "title": result.title.strip() if result.title else "No title available",
        "arXiv_id": arxiv_id,
        "abstract": result.summary.strip() if result.summary else "No abstract available",
        "authors": authors,
        "categories": categories,
        "comments": result.comment or "No comments available",
        "pdf_url": pdf_url,
        "abstract_url": abstract_url,
    }


def get_yesterday_arxiv_papers(category: str = "cs.CV", max_results: int = 100):
    search = _build_search(category, max_results)
    client = arxiv.Client(
        page_size=min(max_results, 100),
        delay_seconds=3,
        num_retries=0,
    )

    for attempt in range(len(ARXIV_TRANSIENT_BACKOFF_SECONDS) + 1):
        try:
            results = list(client.results(search))
            return [_normalize_result(result) for result in results[:max_results]]
        except Exception as error:
            if attempt == len(ARXIV_TRANSIENT_BACKOFF_SECONDS):
                logger.warning(
                    f"Failed to retrieve arXiv papers for {category}; returning empty list. {error}"
                )
                return []
            backoff_seconds = ARXIV_TRANSIENT_BACKOFF_SECONDS[attempt]
            logger.warning(
                f"Transient arXiv error for {category}; retrying in {backoff_seconds}s. {error}"
            )
            time.sleep(backoff_seconds)

    return []


if __name__ == "__main__":
    papers = get_yesterday_arxiv_papers()
    print(len(papers))
