from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from typing import Iterable, Optional

import pdfplumber
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from urllib.parse import urljoin


logger = logging.getLogger("vsrf_raw_scraper")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


@dataclass(frozen=True)
class ScraperConfig:
    base_url: str = "https://vsrf.ru"

    list_url_template: str = (
        "lk/practice/acts?"
        "&numberExact=true"
        "&actDateExact=true"
        "&actDateFrom={act_date}"
        "&caseType=ADMINISTRATIVE_INFRACTION"
        "&actTypeId=11012"
        "{page_part}"
    )

    data_dir: str = "legal-mas/data"
    pdf_dir: str = "legal-mas/data/pdf_documents"
    out_jsonl: str = "legal-mas/data/cases_raw.jsonl"
    out_progress: str = "legal-mas/data/progress.json"

    target_cases: int = 1000
    max_pages_per_day: int = 50

    timeout_sec: int = 20
    max_retries: int = 3
    base_delay_sec: float = 1.5
    jitter_sec: float = 1.0

    download_pdfs: bool = True
    extract_pdf_text: bool = True

class HttpClient:
    """Небольшой HTTP-клиент с ретраями и простой антибот-проверкой."""

    def __init__(self, cfg: ScraperConfig) -> None:
        self.cfg = cfg
        self.session = requests.Session()
        self.headers = {
            "User-Agent": UserAgent().chrome,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "ru-RU,ru;q=0.9",
            "Referer": cfg.base_url,
        }

    def get(self, url: str, stream: bool = False) -> requests.Response:
        last_err: Optional[Exception] = None

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = self.session.get(
                    url,
                    headers=self.headers,
                    timeout=self.cfg.timeout_sec,
                    stream=stream
                )
                resp.raise_for_status()

                if not stream:
                    sample = resp.text[:20000].lower()
                    if "captcha" in sample or "доступ ограничен" in sample:
                        raise RuntimeError("Anti-bot detected (captcha/access restricted)")

                return resp

            except Exception as e:
                last_err = e
                delay = self.cfg.base_delay_sec * attempt + random.random() * self.cfg.jitter_sec
                logger.warning(f"GET failed {attempt}/{self.cfg.max_retries}: {e} | sleep {delay:.1f}s")
                time.sleep(delay)

        raise RuntimeError(f"GET failed after retries: {url}") from last_err


@dataclass
class CaseStub:
    case_id: str
    list_date_raw: str
    case_page_url: Optional[str]
    pdf_url: Optional[str]
    doc_type: Optional[str]


def parse_case_list_page(html: str, base_url: str) -> list[CaseStub]:
    """Разбирает страницу выдачи в список коротких карточек дел."""
    soup = BeautifulSoup(html, "html.parser")
    containers = soup.select(".vs-items.vs-list-items > .vs-items-body")

    out: list[CaseStub] = []
    for c in containers:
        a = c.select_one(".vs-items-label a")
        case_id = a.get_text(strip=True) if a else ""
        case_href = a.get("href") if a else None
        case_page_url = urljoin(base_url, case_href) if case_href else None

        list_date_raw = ""
        d = c.select_one(".col-md-3.vs-font-20")
        if d:
            list_date_raw = d.get_text(strip=True)

        doc_type = None
        t = c.select_one(".vs-collapse a")
        if t:
            doc_type = t.get_text(strip=True) or None

        pdf_url = None
        pdf_a = c.select_one("a[download]")
        if pdf_a:
            pdf_href = pdf_a.get("href")
            if pdf_href:
                pdf_url = urljoin(base_url, pdf_href)

        if case_id:
            out.append(CaseStub(
                case_id=case_id,
                list_date_raw=list_date_raw,
                case_page_url=case_page_url,
                pdf_url=pdf_url,
                doc_type=doc_type
            ))

    return out


def parse_articles_from_case_page(html: str) -> list[str]:
    """Извлекает упоминания статей со страницы карточки дела."""
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("div", class_="row vs-item-detail")
    articles: list[str] = []

    for row in rows:
        label = row.find("div", class_="col-md-3")
        if not label:
            continue

        label_text = label.get_text(strip=True)
        if "Статьи КоАП" not in label_text and "Статьи" not in label_text:
            continue

        value = row.find("div", class_="col-md-7 vs-items-additional-info")
        if not value:
            continue

        txt = value.get_text(strip=True).replace("КоАП:", "").strip()
        for m in re.finditer(r"(\d+(?:\.\d+)?)(?:\s*ч\.?\s*(\d+))?", txt):
            art = m.group(1)
            part = m.group(2)
            if part:
                articles.append(f"ст. {art} ч.{part} КоАП РФ")
            else:
                articles.append(f"ст. {art} КоАП РФ")

    seen = set()
    uniq = []
    for a in articles:
        if a not in seen:
            uniq.append(a)
            seen.add(a)
    return uniq

def save_pdf_bytes(pdf_bytes: bytes, pdf_path: str) -> None:
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Собирает текст со всех страниц PDF."""
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        parts = []
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                parts.append(t)
        return "\n".join(parts)

def load_progress(path: str) -> dict:
    if not os.path.exists(path):
        return {"seen_case_ids": [], "written": 0, "last_day": None}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_progress(path: str, progress: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, items: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            n += 1
    return n

class VSRFRawScraper:
    """Собирает сырые данные дел: метаданные, PDF и извлеченный текст."""

    def __init__(self, cfg: ScraperConfig) -> None:
        self.cfg = cfg
        self.http = HttpClient(cfg)
        os.makedirs(cfg.data_dir, exist_ok=True)
        os.makedirs(cfg.pdf_dir, exist_ok=True)

    def build_list_url(self, act_date: str, page: int) -> str:
        page_part = f"&page={page}" if page > 1 else ""
        rel = self.cfg.list_url_template.format(act_date=act_date, page_part=page_part)
        return urljoin(self.cfg.base_url, rel)

    def fetch_html(self, url: str) -> str:
        return self.http.get(url, stream=False).text

    def fetch_pdf_bytes(self, url: str) -> bytes:
        return self.http.get(url, stream=True).content

    def scrape_day(self, act_date: str, seen_ids: set[str]) -> list[dict]:
        """Собирает сырые записи за одну дату по всем страницам выдачи."""
        results: list[dict] = []

        for page in range(1, self.cfg.max_pages_per_day + 1):
            list_url = self.build_list_url(act_date, page)
            logger.info(f"List page: {list_url}")

            try:
                html = self.fetch_html(list_url)
            except Exception as e:
                logger.warning(f"Skip list page due to error: {e}")
                break

            stubs = parse_case_list_page(html, self.cfg.base_url)
            if not stubs:
                break

            for s in stubs:
                if s.case_id in seen_ids:
                    continue

                record = {
                    "case_id": s.case_id,
                    "source": "vsrf",
                    "source_url": s.case_page_url,
                    "pdf_url": s.pdf_url,
                    "list_date_raw": s.list_date_raw,
                    "document_type": s.doc_type,
                    "site_meta": {
                        "articles": [],
                    },
                    "pdf_path": None,
                    "full_text": None,
                    "created_at": datetime.utcnow().isoformat(),
                }

                if s.case_page_url:
                    try:
                        case_html = self.fetch_html(s.case_page_url)
                        record["site_meta"]["articles"] = parse_articles_from_case_page(case_html)
                    except Exception as e:
                        logger.warning(f"Case page meta failed {s.case_id}: {e}")

                if s.pdf_url:
                    try:
                        pdf_bytes = self.fetch_pdf_bytes(s.pdf_url)

                        if self.cfg.download_pdfs:
                            pdf_path = os.path.join(self.cfg.pdf_dir, f"{s.case_id}.pdf")
                            save_pdf_bytes(pdf_bytes, pdf_path)
                            record["pdf_path"] = pdf_path

                        if self.cfg.extract_pdf_text:
                            record["full_text"] = extract_text_from_pdf_bytes(pdf_bytes)

                    except Exception as e:
                        logger.warning(f"PDF failed {s.case_id}: {e}")

                results.append(record)
                seen_ids.add(s.case_id)

            time.sleep(self.cfg.base_delay_sec + random.random() * self.cfg.jitter_sec)

        return results


def main():
    cfg = ScraperConfig()

    progress = load_progress(cfg.out_progress)
    seen_ids = set(progress.get("seen_case_ids", []))
    collected = int(progress.get("written", 0))

    scraper = VSRFRawScraper(cfg)
    current = date.today()

    while collected < cfg.target_cases:
        act_date = current.strftime("%d.%m.%Y")
        logger.info(f"Scrape day: {act_date} | collected={collected}/{cfg.target_cases}")

        day_records = scraper.scrape_day(act_date, seen_ids)
        if day_records:
            written = append_jsonl(cfg.out_jsonl, day_records)
            collected += written

            progress["written"] = collected
            progress["seen_case_ids"] = list(seen_ids)
            progress["last_day"] = act_date
            save_progress(cfg.out_progress, progress)

            logger.info(f"Written {written}, total {collected}")
        else:
            logger.info("No records on this day.")

        current = date.fromordinal(current.toordinal() - 1)
        time.sleep(cfg.base_delay_sec * 2 + random.random() * cfg.jitter_sec)

    logger.info("Done.")


if __name__ == "__main__":
    main()
