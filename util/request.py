"""
Use requests and BeautifulSoup to get yesterday's arXiv papers.
"""

import requests
import re
import time
from bs4 import BeautifulSoup
from loguru import logger


ARXIV_ALLOWED_SHOW_VALUES = (25, 50, 100, 250, 500, 1000, 2000)
ARXIV_RATE_LIMIT_BACKOFF_SECONDS = (30, 60, 120)


def get_arxiv_show_value(max_results: int):
    for show_value in ARXIV_ALLOWED_SHOW_VALUES:
        if max_results <= show_value:
            return show_value
    return ARXIV_ALLOWED_SHOW_VALUES[-1]


def get_yesterday_arxiv_papers(category: str = "cs.CV", max_results: int = 100):
    show_value = get_arxiv_show_value(max_results)
    url = f"https://arxiv.org/list/{category}/new?skip=0&show={show_value}"

    response = None
    for attempt in range(len(ARXIV_RATE_LIMIT_BACKOFF_SECONDS) + 1):
        response = requests.get(url)
        if response.status_code != 429:
            break
        logger.warning("arXiv rate limited; try later.")
        if attempt == len(ARXIV_RATE_LIMIT_BACKOFF_SECONDS):
            return []
        time.sleep(ARXIV_RATE_LIMIT_BACKOFF_SECONDS[attempt])

    if response is None or response.status_code == 429:
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    try:
        entries = soup.find_all("dl", id="articles")[0].find_all(["dt", "dd"])
    except Exception as e:
        return []

    papers = []
    for i in range(0, len(entries), 2):
        title_tag = entries[i + 1].find("div", class_="list-title")
        title = (
            title_tag.text.strip().replace("Title:", "").strip()
            if title_tag
            else "No title available"
        )

        abs_url = "https://arxiv.org" + entries[i].find("a", title="Abstract")["href"]

        pdf_url = entries[i].find("a", title="Download PDF")["href"]
        pdf_url = "https://arxiv.org" + pdf_url

        abstract_tag = entries[i + 1].find("p", class_="mathjax")
        abstract = (
            abstract_tag.text.strip() if abstract_tag else "No abstract available"
        )

        authors_tag = entries[i + 1].find("div", class_="list-authors")
        authors = (
            [
                author.text.strip()
                for author in authors_tag.find_all("a")
                if author.text.strip()
            ]
            if authors_tag
            else []
        )

        subjects_tag = entries[i + 1].find("div", class_="list-subjects")
        categories = []
        if subjects_tag:
            subjects_text = subjects_tag.get_text(" ", strip=True).replace(
                "Subjects:", ""
            )
            categories = re.findall(r"\(([^)]+)\)", subjects_text)
            if not categories:
                primary_subject = subjects_tag.find("span", class_="primary-subject")
                if primary_subject:
                    categories = [primary_subject.text.strip()]

        comments_tag = entries[i + 1].find("div", class_="list-comments")
        comments = (
            comments_tag.text.strip() if comments_tag else "No comments available"
        )

        paper_info = {
            "title": title,
            "arXiv_id": pdf_url.split("/")[-1],
            "abstract": abstract,
            "authors": authors,
            "categories": categories,
            "comments": comments,
            "pdf_url": pdf_url,
            "abstract_url": abs_url,
        }

        papers.append(paper_info)

    return papers[:max_results]


if __name__ == "__main__":
    papers = get_yesterday_arxiv_papers()
    print(len(papers))
