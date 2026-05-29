from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WebProbeCase:
    id: str
    url: str
    question: str
    must_include: tuple[str, ...]
    accepted_terms: tuple[str, ...]


DEFAULT_WEB_PROBES = (
    WebProbeCase(
        id="web_bbc_news_001",
        url="https://www.bbc.com/news",
        question="What kind of website is shown, and what is it about?",
        must_include=("BBC", "News"),
        accepted_terms=("news", "headlines", "stories", "articles"),
    ),
    WebProbeCase(
        id="web_hacker_news_002",
        url="https://news.ycombinator.com/",
        question="What kind of page is this and what content is listed?",
        must_include=("Hacker News",),
        accepted_terms=("news", "links", "stories", "comments"),
    ),
    WebProbeCase(
        id="web_github_trending_003",
        url="https://github.com/trending",
        question="What developer website is shown and what is the page listing?",
        must_include=("GitHub", "Trending"),
        accepted_terms=("repositories", "developers", "trending", "code"),
    ),
    WebProbeCase(
        id="web_nasa_004",
        url="https://www.nasa.gov/",
        question="What organization is this webpage for and what type of content is visible?",
        must_include=("NASA",),
        accepted_terms=("space", "science", "missions", "news"),
    ),
    WebProbeCase(
        id="web_wikipedia_rover_005",
        url="https://en.wikipedia.org/wiki/Mars_rover",
        question="What topic is this article about?",
        must_include=("Mars rover",),
        accepted_terms=("mars", "rover", "spacecraft", "planet"),
    ),
)


async def generate_web_probe(output_dir: Path, limit: int = 5, width: int = 1920, height: int = 1080) -> Path:
    if limit <= 0:
        raise ValueError("limit must be positive")
    cases = DEFAULT_WEB_PROBES[:limit]
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Install web probe dependencies with `python -m pip install -e '.[web]'` "
            "and `python -m playwright install chromium`."
        ) from exc

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": width, "height": height})
        with manifest_path.open("w", encoding="utf-8") as handle:
            for index, case in enumerate(cases, start=1):
                await page.goto(case.url, wait_until="domcontentloaded", timeout=45_000)
                await page.wait_for_timeout(1_000)
                image_name = f"{index:03d}.png"
                await page.screenshot(path=str(images_dir / image_name), full_page=False)
                ocr_text = await _page_text(page)
                handle.write(
                    json.dumps(
                        {
                            "id": case.id,
                            "image": f"images/{image_name}",
                            "source_url": case.url,
                            "question": case.question,
                            "ocr_text": ocr_text[:4000],
                            "rubric": "description",
                            "must_include": case.must_include,
                            "accepted_fix_terms": case.accepted_terms,
                            "must_not_include": (),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        await browser.close()
    return manifest_path


async def _page_text(page) -> str:
    title = await page.title()
    body = await page.locator("body").inner_text(timeout=10_000)
    return f"{title}\n{body}"
