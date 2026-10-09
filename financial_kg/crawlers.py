"""The original Wikipedia and Investopedia extraction strategies."""

import csv
import time
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup


HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FinancialKGBuilder/1.0)"}


def fetch_soup(url):
    response = requests.get(
        url.replace("&apos;", "%27"), timeout=15, headers=HEADERS
    )
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


def wiki_extract(url, keyword):
    soup = fetch_soup(url)
    main = soup.find("div", class_="mw-parser-output")
    content = "\n".join(p.get_text(" ", strip=True) for p in main.find_all("p")) if main else ""
    related = [ul.get_text(" ", strip=True) for ul in soup.find_all("ul")]
    return {
        "Entity": keyword,
        "Content": content,
        "Related_Entities": [item for item in related if 4 < len(item) < 15],
    }


def investopedia_extract(url, keyword):
    soup = fetch_soup(url)
    title = soup.find("title")
    article = soup.find("article")
    if article is None:
        article = soup.find("div", class_="content")
    content = (
        article.get_text(" ", strip=True)
        if article is not None
        else " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    )
    return {
        "Entity": keyword,
        "Title": title.get_text(strip=True) if title else "",
        "Content": content,
    }


def crawl_sources(path, delay=5, limit=None):
    """Yield articles in CSV order; both crawlers return Content."""
    extractors = {"wiki": wiki_extract, "investopedia": investopedia_extract}
    with open(path, encoding="utf-8-sig", newline="") as source_file:
        for index, row in enumerate(csv.DictReader(source_file)):
            if limit is not None and index >= limit:
                break
            source, url = row["source"].strip(), row["URL"].strip()
            if source not in extractors:
                raise ValueError(f"Unsupported source: {source}")
            if index and delay:
                time.sleep(delay)
            entity = unquote(urlparse(url).path.rstrip("/").split("/")[-1])
            if entity.endswith(".asp"):
                entity = entity[:-4]
            article = extractors[source](url, entity)
            article.update(Source=source, URL=url)
            yield article
