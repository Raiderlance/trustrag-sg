"""Download, extract, and chunk authoritative HTML sources for RAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from html import escape as escape_html
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
from lxml import html

LOGGER = logging.getLogger("ingest")
ALLOWED_HOSTS = {"www.hdb.gov.sg", "www.cpf.gov.sg"}
BLOCK_TAGS = {"p", "li", "blockquote", "pre", "table"}
CONTENT_XPATHS = (
    "//*[contains(concat(' ', normalize-space(@class), ' '), ' faq ')][.//lightning-formatted-rich-text]",
    "//main",
    "//*[@role='main']",
    "//*[contains(concat(' ', normalize-space(@class), ' '), ' cmp-container ')]",
    "//*[contains(concat(' ', normalize-space(@class), ' '), ' content ')]",
    "//article",
)
DROP_XPATH = (
    ".//script | .//style | .//noscript | .//svg | .//form | .//nav | "
    ".//header | .//footer | .//aside | .//*[@aria-hidden='true'] | "
    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' breadcrumb ')] | "
    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' cookie ')]"
)


@dataclass(frozen=True)
class Section:
    """Extracted document section with its heading hierarchy and text."""
    heading_path: tuple[str, ...]
    text: str


def clean_text(value: str) -> str:
    """Collapse repeated whitespace and trim extracted page text."""
    value = value.replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def extract_next_data_article(tree: html.HtmlElement) -> html.HtmlElement | None:
    """Return article markup embedded by Next.js, as used by saved HDB pages."""
    nodes = tree.xpath("//script[@id='__NEXT_DATA__']")
    if not nodes or not nodes[0].text:
        return None
    try:
        payload = json.loads(nodes[0].text)
    except json.JSONDecodeError:
        return None

    articles: list[str] = []

    def walk(value: object) -> None:
        """Recursively locate candidate article bodies in Next.js data."""
        if isinstance(value, dict):
            body = value.get("bodyContent")
            if isinstance(body, dict) and isinstance(body.get("value"), str):
                articles.append(body["value"])
            # HDB stores accordion panels separately from the main body.
            accordion_body = value.get("bodyContentVal")
            if isinstance(accordion_body, str):
                title = next(
                    (value.get(key) for key in ("title", "heading", "header", "label")
                     if isinstance(value.get(key), str)),
                    None,
                )
                articles.append((f"<h2>{escape_html(title)}</h2>" if title else "") + accordion_body)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    if not articles:
        return None
    # Keep document order and avoid duplicate CMS payloads.
    markup = "\n".join(dict.fromkeys(articles))
    return html.fragment_fromstring(markup, create_parent="main")


def load_manifest(path: Path) -> list[dict]:
    """Load and validate unique source records from the JSON manifest."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Manifest root must be a JSON array")
    required = {"source_id", "agency", "title", "url", "domain", "source_type"}
    seen: set[str] = set()
    for index, item in enumerate(data):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Manifest item {index} lacks: {sorted(missing)}")
        if item["source_id"] in seen:
            raise ValueError(f"Duplicate source_id: {item['source_id']}")
        seen.add(item["source_id"])
        if item["source_type"] != "html":
            raise ValueError(f"Unsupported source_type for {item['source_id']}")
        if urlparse(item["url"]).hostname not in ALLOWED_HOSTS:
            raise ValueError(f"Host is not allow-listed: {item['url']}")
    return data


def download(source: dict, raw_dir: Path, session: requests.Session, timeout: int) -> Path:
    """Download one source page and save its raw HTML response."""
    target = raw_dir / f"{source['source_id']}.html"
    response = session.get(source["url"], timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "html" not in content_type:
        raise ValueError(f"Expected HTML, received {content_type!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return target


def render_download(source: dict, raw_dir: Path, timeout: int) -> Path:
    """Render a JavaScript page in Chromium and save its hydrated DOM."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for rendered sources; install requirements and run "
            "'python -m playwright install chromium'"
        ) from exc

    target = raw_dir / f"{source['source_id']}.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    timeout_ms = timeout * 1000
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(source["url"], wait_until="domcontentloaded", timeout=timeout_ms)
            # Network-idle is not reliable on analytics-heavy pages. Wait for the
            # document to gain substantial visible text, then allow hydration to settle.
            page.wait_for_function(
                "() => document.body && document.body.innerText.trim().length > 500",
                timeout=timeout_ms,
            )
            page.wait_for_timeout(2000)
            target.write_text(page.content(), encoding="utf-8")
        finally:
            browser.close()
    return target


def select_content(tree: html.HtmlElement) -> html.HtmlElement:
    """Choose the most likely main-content element from a parsed page."""
    for xpath in CONTENT_XPATHS:
        candidates = tree.xpath(xpath)
        if candidates:
            candidate = max(candidates, key=lambda node: len(clean_text(node.text_content())))
            if clean_text(candidate.text_content()):
                return candidate
    body = tree.find("body")
    return body if body is not None else tree


def extract_sections(raw_html: bytes) -> tuple[str | None, list[Section]]:
    """Extract a canonical URL and heading-aware text sections from HTML."""
    parser = html.HTMLParser(encoding="utf-8", recover=True)
    tree = html.fromstring(raw_html, parser=parser)
    canonical = tree.xpath("string(//link[@rel='canonical']/@href)").strip() or None
    embedded_root = extract_next_data_article(tree)
    root = embedded_root if embedded_root is not None else select_content(tree)
    for node in root.xpath(DROP_XPATH):
        parent = node.getparent()
        if parent is not None:
            parent.remove(node)

    headings: dict[int, str] = {}
    sections: list[Section] = []
    buffer: list[str] = []

    def flush() -> None:
        """Store accumulated text as a section when it is non-empty."""
        text = clean_text(" ".join(buffer))
        if text:
            path = tuple(headings[level] for level in sorted(headings))
            sections.append(Section(path, text))
        buffer.clear()

    for node in root.iter():
        tag = node.tag.lower() if isinstance(node.tag, str) else ""
        if re.fullmatch(r"h[1-6]", tag):
            flush()
            level = int(tag[1])
            heading = clean_text(node.text_content())
            if heading:
                headings[level] = heading
                headings = {k: v for k, v in headings.items() if k <= level}
        elif tag in BLOCK_TAGS:
            # Tables retain row/cell boundaries as readable separators.
            if tag == "table":
                rows = []
                for row in node.xpath(".//tr"):
                    cells = [clean_text(c.text_content()) for c in row.xpath("./th|./td")]
                    if any(cells):
                        rows.append(" | ".join(cells))
                value = " ; ".join(rows)
            else:
                value = clean_text(node.text_content())
            if value:
                buffer.append(value)
        elif tag == "div" and node.xpath("ancestor::lightning-formatted-rich-text"):
            # Salesforce LWR emits rich-text paragraphs as leaf divs.
            has_block_children = bool(node.xpath(".//p | .//li | .//table | .//h1 | .//h2 | .//h3 | .//h4 | .//h5 | .//h6"))
            if not has_block_children:
                value = clean_text(node.text_content())
                if value:
                    buffer.append(value)
    flush()
    # Some CMS pages render meaningful copy directly in divs rather than p/li tags.
    # Preserve their text as one section instead of silently emitting no content.
    if not sections:
        fallback = clean_text(root.text_content())
        if fallback:
            sections.append(Section((), fallback))
    return canonical, sections


def split_words(text: str, max_words: int, overlap_words: int) -> Iterable[str]:
    """Yield fixed-size word windows with the requested adjacent overlap."""
    words = text.split()
    if not words:
        return
    step = max_words - overlap_words
    for start in range(0, len(words), step):
        chunk = words[start : start + max_words]
        if chunk:
            yield " ".join(chunk)
        if start + max_words >= len(words):
            break


def make_chunks(source: dict, canonical: str | None, sections: list[Section], max_words: int,
                overlap_words: int, content_hash: str) -> list[dict]:
    """Convert extracted sections into chunk records with source metadata."""
    records = []
    for section_index, section in enumerate(sections):
        for part_index, text in enumerate(split_words(section.text, max_words, overlap_words)):
            chunk_id = f"{source['source_id']}:{section_index:03d}:{part_index:03d}"
            records.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": {
                    "source_id": source["source_id"],
                    "agency": source["agency"],
                    "title": source["title"],
                    "url": canonical or source["url"],
                    "manifest_url": source["url"],
                    "domain": source["domain"],
                    "priority": source.get("priority"),
                    "retrieved_at": source.get("retrieved_at"),
                    "heading_path": list(section.heading_path),
                    "section_index": section_index,
                    "part_index": part_index,
                    "content_sha256": content_hash,
                },
            })
    return records


def main() -> int:
    """Download configured sources, extract chunks, and write ingestion reports."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Processed output directory (default: <data-dir>/processed)",
    )
    parser.add_argument("--use-cache", action="store_true", help="Do not redownload existing HTML")
    parser.add_argument(
        "--render-source",
        action="append",
        default=[],
        metavar="SOURCE_ID",
        help="Render this source with Playwright (repeatable; overrides its cache)",
    )
    parser.add_argument("--max-words", type=int, default=350)
    parser.add_argument("--overlap-words", type=int, default=50)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    if args.max_words <= 0 or not 0 <= args.overlap_words < args.max_words:
        parser.error("Require max-words > 0 and 0 <= overlap-words < max-words")

    sources = load_manifest(args.manifest)
    known_source_ids = {source["source_id"] for source in sources}
    rendered_source_ids = set(args.render_source)
    unknown_rendered = rendered_source_ids - known_source_ids
    if unknown_rendered:
        parser.error(f"Unknown --render-source IDs: {sorted(unknown_rendered)}")
    raw_dir = args.data_dir / "raw"
    output_dir = args.output_dir or args.data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "TrustRAG-SG/0.1 (research prototype; contact project owner)"
    all_chunks, report = [], []

    for index, source in enumerate(sources):
        try:
            raw_path = raw_dir / f"{source['source_id']}.html"
            if source["source_id"] in rendered_source_ids:
                raw_path = render_download(source, raw_dir, args.timeout)
            elif not (args.use_cache and raw_path.exists()):
                raw_path = download(source, raw_dir, session, args.timeout)
                if index < len(sources) - 1:
                    time.sleep(args.delay)
            raw = raw_path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            canonical, sections = extract_sections(raw)
            chunks = make_chunks(source, canonical, sections, args.max_words, args.overlap_words, digest)
            if not chunks:
                raise ValueError("No content chunks extracted")
            all_chunks.extend(chunks)
            report.append({"source_id": source["source_id"], "status": "ok", "sections": len(sections), "chunks": len(chunks), "sha256": digest})
            LOGGER.info("%s: %d sections, %d chunks", source["source_id"], len(sections), len(chunks))
        except Exception as exc:  # Continue so one changed page does not lose the batch.
            LOGGER.exception("Failed %s", source["source_id"])
            report.append({"source_id": source["source_id"], "status": "error", "error": str(exc)})

    chunks_path = output_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in all_chunks:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    run_report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "chunking": {
            "max_words": args.max_words,
            "overlap_words": args.overlap_words,
        },
        "sources": report,
        "total_chunks": len(all_chunks),
    }
    (output_dir / "ingestion_report.json").write_text(
        json.dumps(run_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 1 if any(item["status"] == "error" for item in report) else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
