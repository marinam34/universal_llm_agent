from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from core.config import get_settings
from core.models import (
    ButtonDescriptor,
    FieldDescriptor,
    FormDescriptor,
    LinkDescriptor,
    PageSnapshot,
    SiteKnowledgeBase,
    TransitionEdge,
)


TAG_KEYWORDS = {
    "doctors": ("врач", "doctor", "специалист"),
    "clinics": ("клиник", "clinic", "центр", "hospital", "стационар"),
    "services": ("услуг", "service", "направлен"),
    "analyses": ("анализ", "lab", "smartlab"),
    "booking": ("запис", "appointment", "book", "заявк", "обратный звонок", "call"),
    "contacts": ("контакт", "адрес", "телефон", "phone"),
    "prices": ("цен", "price", "стоим"),
    "faq": ("вопрос", "faq", "помощ"),
    "children": ("дет", "kids"),
    "beauty": ("beauty", "космет", "пластическ"),
}


def _base_domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return parsed._replace(fragment="", query="", path=path).geturl()


def _same_site(url: str, site_root: str) -> bool:
    return urlparse(url).netloc.lower() == urlparse(site_root).netloc.lower()


def _short_text(text: str | None, limit: int = 200) -> str:
    return (text or "").strip().replace("\n", " ")[:limit]


def _normalized_target(page_url: str, raw_href: str | None) -> str | None:
    href = (raw_href or "").strip()
    if not href:
        return None
    lowered = href.lower()
    if lowered.startswith(("#", "javascript:", "mailto:", "tel:")):
        return None

    target = _normalize_url(urljoin(page_url, href))
    if target == _normalize_url(page_url):
        return None
    return target


def _page_slug(url: str) -> str:
    parsed = urlparse(url)
    raw = (parsed.path or "root").strip("/") or "root"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw)
    return slug[:120]


def _stable_selector(tag) -> str:
    if not tag:
        return ""
    if tag.get("id"):
        return f"#{tag['id']}"
    if tag.get("data-testid"):
        return f'{tag.name}[data-testid="{tag["data-testid"]}"]'
    if tag.get("name"):
        return f'{tag.name}[name="{tag["name"]}"]'
    if tag.get("aria-label"):
        return f'{tag.name}[aria-label="{tag["aria-label"]}"]'
    if tag.name == "a" and tag.get("href"):
        return f'a[href="{tag.get("href")}"]'
    if tag.get("placeholder"):
        return f'{tag.name}[placeholder="{tag["placeholder"]}"]'
    if tag.get("type") and tag.get("value"):
        return f'{tag.name}[type="{tag["type"]}"][value="{tag["value"]}"]'

    parts: list[str] = []
    current = tag
    depth = 0
    while current and getattr(current, "name", None) and depth < 4:
        segment = current.name
        siblings = [sib for sib in current.parent.find_all(current.name, recursive=False)] if current.parent else []
        if len(siblings) > 1:
            position = siblings.index(current) + 1
            segment += f":nth-of-type({position})"
        parts.append(segment)
        current = current.parent
        depth += 1
    return " > ".join(reversed(parts))


def _label_for_field(soup: BeautifulSoup, field) -> str | None:
    field_id = field.get("id")
    if field_id:
        label = soup.find("label", attrs={"for": field_id})
        if label and label.get_text(strip=True):
            return _short_text(label.get_text(" ", strip=True))

    parent_label = field.find_parent("label")
    if parent_label and parent_label.get_text(strip=True):
        return _short_text(parent_label.get_text(" ", strip=True))

    previous = field.find_previous(string=True)
    if previous:
        candidate = _short_text(str(previous))
        if candidate and len(candidate) <= 80:
            return candidate

    return field.get("aria-label") or field.get("placeholder")


def _button_target(page_url: str, button) -> str | None:
    parent_link = button.find_parent("a", href=True)
    if parent_link:
        return _normalized_target(page_url, parent_link.get("href", ""))

    for attr in ("formaction", "data-href", "data-url", "data-link", "data-target-url", "data-target-href"):
        if button.get(attr):
            return _normalized_target(page_url, button.get(attr, ""))

    onclick = button.get("onclick") or ""
    match = re.search(r"""(?:location(?:\.href)?|window\.open)\s*[\(=]\s*['"]([^'"]+)['"]""", onclick)
    if match:
        return _normalized_target(page_url, match.group(1))

    parent_form = button.find_parent("form")
    if parent_form and parent_form.get("action"):
        return _normalized_target(page_url, parent_form.get("action", ""))

    return None


def _page_tags(page_url: str, title: str | None, headings: list[str], visible_text_sample: list[str], forms: list[FormDescriptor]) -> list[str]:
    haystack = " ".join([page_url, title or "", " ".join(headings[:6]), " ".join(visible_text_sample[:10])]).lower()
    tags = {tag for tag, patterns in TAG_KEYWORDS.items() if any(pattern in haystack for pattern in patterns)}
    if forms:
        tags.add("has_form")
    if any(keyword in haystack for keyword in ("заявк", "callback", "call", "отправ")):
        tags.add("lead_form")
    return sorted(tags)


def _page_summary(
    page_url: str,
    title: str | None,
    tags: list[str],
    headings: list[str],
    links: list[LinkDescriptor],
    buttons: list[ButtonDescriptor],
    forms: list[FormDescriptor],
    inputs: list[FieldDescriptor],
    visible_text_sample: list[str],
) -> str:
    parts: list[str] = []
    if title:
        parts.append(f"title={title}")
    if tags:
        parts.append(f"tags={', '.join(tags[:8])}")
    if headings:
        parts.append(f"headings={'; '.join(headings[:4])}")
    key_links = [link.text for link in links[:8] if link.text]
    if key_links:
        parts.append(f"links={'; '.join(key_links[:6])}")
    key_buttons = [button.text for button in buttons[:8] if button.text]
    if key_buttons:
        parts.append(f"buttons={'; '.join(key_buttons[:6])}")
    if forms:
        form_fields = []
        for field in forms[0].fields[:8]:
            form_fields.append(field.label or field.name or field.selector)
        if form_fields:
            parts.append(f"form_fields={'; '.join(form_fields)}")
    elif inputs:
        field_labels = [field.label or field.name or field.selector for field in inputs[:8]]
        parts.append(f"inputs={'; '.join(field_labels)}")
    if visible_text_sample:
        parts.append(f"text={'; '.join(visible_text_sample[:3])}")
    if not parts:
        parts.append(page_url)
    return " | ".join(parts)[:1600]


def _dedupe_links(links: list[LinkDescriptor]) -> list[LinkDescriptor]:
    deduped: list[LinkDescriptor] = []
    seen: set[tuple[str, str]] = set()
    for link in links:
        key = ((link.href or "").strip(), (link.text or "").strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(link)
    return deduped


def _dedupe_buttons(buttons: list[ButtonDescriptor]) -> list[ButtonDescriptor]:
    deduped: list[ButtonDescriptor] = []
    seen: set[tuple[str, str, str]] = set()
    for button in buttons:
        key = ((button.target_url or "").strip(), (button.text or "").strip(), (button.selector or "").strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(button)
    return deduped


def _dedupe_inputs(inputs: list[FieldDescriptor]) -> list[FieldDescriptor]:
    deduped: list[FieldDescriptor] = []
    seen: set[str] = set()
    for field in inputs:
        key = (field.selector or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(field)
    return deduped


def _dedupe_transitions(transitions: list[TransitionEdge]) -> list[TransitionEdge]:
    deduped: list[TransitionEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in transitions:
        key = (edge.from_url, edge.to_url, edge.trigger_type)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def _extract_page(page_url: str, html: str, preview_image_url: str | None) -> tuple[PageSnapshot, list[TransitionEdge]]:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else None
    headings = [_short_text(h.get_text(" ", strip=True)) for h in soup.select("h1, h2, h3") if h.get_text(strip=True)]
    visible_text_sample = [
        _short_text(text, 240)
        for text in soup.stripped_strings
        if text and len(text.strip()) > 2
    ][:25]

    links: list[LinkDescriptor] = []
    buttons: list[ButtonDescriptor] = []
    forms: list[FormDescriptor] = []
    inputs: list[FieldDescriptor] = []
    transitions: list[TransitionEdge] = []

    for link in soup.select("a[href]"):
        href = _normalized_target(page_url, link.get("href", ""))
        if not href:
            continue
        text = _short_text(link.get_text(" ", strip=True) or link.get("aria-label") or href)
        selector = _stable_selector(link)
        links.append(
            LinkDescriptor(
                text=text,
                selector=selector,
                href=href,
                title=_short_text(link.get("title")),
            )
        )
        transitions.append(
            TransitionEdge(
                from_url=page_url,
                to_url=href,
                trigger_text=text,
                selector=selector,
                trigger_type="link",
            )
        )

    for button in soup.select("button, input[type=button], input[type=submit], [role=button]"):
        text = _short_text(button.get_text(" ", strip=True) or button.get("value") or button.get("aria-label"))
        selector = _stable_selector(button)
        target_url = _button_target(page_url, button)
        buttons.append(
            ButtonDescriptor(
                text=text,
                selector=selector,
                button_type=button.get("type"),
                target_url=target_url,
                title=_short_text(button.get("title")),
            )
        )
        if target_url:
            transitions.append(
                TransitionEdge(
                    from_url=page_url,
                    to_url=target_url,
                    trigger_text=text,
                    selector=selector,
                    trigger_type="button",
                )
            )

    for field in soup.select("input, textarea, select"):
        inputs.append(
            FieldDescriptor(
                selector=_stable_selector(field),
                name=field.get("name"),
                input_type=field.get("type") or field.name,
                label=_label_for_field(soup, field),
                placeholder=field.get("placeholder"),
                required=field.has_attr("required") or field.get("aria-required") == "true",
                options=[_short_text(opt.get_text(" ", strip=True)) for opt in field.select("option") if opt.get_text(strip=True)],
            )
        )

    for form in soup.select("form"):
        form_action = _normalized_target(page_url, form.get("action", "")) if form.get("action") else None
        form_fields: list[FieldDescriptor] = []
        form_buttons: list[ButtonDescriptor] = []
        for field in form.select("input, textarea, select"):
            form_fields.append(
                FieldDescriptor(
                    selector=_stable_selector(field),
                    name=field.get("name"),
                    input_type=field.get("type") or field.name,
                    label=_label_for_field(soup, field),
                    placeholder=field.get("placeholder"),
                    required=field.has_attr("required") or field.get("aria-required") == "true",
                    options=[_short_text(opt.get_text(" ", strip=True)) for opt in field.select("option") if opt.get_text(strip=True)],
                )
            )
        for button in form.select("button, input[type=button], input[type=submit], [role=button]"):
            text = _short_text(button.get_text(" ", strip=True) or button.get("value") or button.get("aria-label"))
            form_buttons.append(
                ButtonDescriptor(
                    text=text,
                    selector=_stable_selector(button),
                    button_type=button.get("type"),
                    target_url=_button_target(page_url, button) or form_action,
                    title=_short_text(button.get("title")),
                )
            )
        forms.append(
            FormDescriptor(
                selector=_stable_selector(form),
                action=form_action,
                method=form.get("method"),
                fields=_dedupe_inputs(form_fields),
                buttons=_dedupe_buttons(form_buttons),
            )
        )
        if form_action:
            transitions.append(
                TransitionEdge(
                    from_url=page_url,
                    to_url=form_action,
                    trigger_text=_short_text(" / ".join(button.text for button in form_buttons if button.text)),
                    selector=_stable_selector(form),
                    trigger_type="form",
                )
            )

    links = _dedupe_links(links)
    buttons = _dedupe_buttons(buttons)
    inputs = _dedupe_inputs(inputs)
    transitions = _dedupe_transitions(transitions)
    text_content = " ".join(soup.stripped_strings)
    tags = _page_tags(page_url, title, headings, visible_text_sample, forms)
    summary = _page_summary(page_url, title, tags, headings, links, buttons, forms, inputs, visible_text_sample)
    text_excerpt = text_content[:3000]

    snapshot = PageSnapshot(
        url=page_url,
        title=title,
        summary=summary,
        tags=tags,
        headings=headings[:24],
        links=links[:300],
        buttons=buttons[:220],
        forms=forms[:60],
        inputs=inputs[:320],
        text_content=text_content[:25000],
        text_excerpt=text_excerpt,
        visible_text_sample=visible_text_sample,
        preview_image_url=preview_image_url,
    )
    return snapshot, transitions


class SiteCrawler:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _site_dir(self, site_root: str) -> Path:
        site_dir = self.settings.artifacts_dir / urlparse(site_root).netloc.replace(":", "_")
        site_dir.mkdir(parents=True, exist_ok=True)
        return site_dir

    def _preview_path(self, site_root: str, page_url: str) -> tuple[Path, str]:
        previews_dir = self._site_dir(site_root) / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_page_slug(page_url)}.png"
        file_path = previews_dir / filename
        web_path = f"/artifacts/{urlparse(site_root).netloc.replace(':', '_')}/previews/{filename}"
        return file_path, web_path

    async def _new_context(self, browser: Browser) -> BrowserContext:
        return await browser.new_context(
            user_agent=self.settings.site_user_agent,
            viewport={"width": 1440, "height": 960},
        )

    async def _prepare_page(self, page: Page) -> None:
        try:
            await page.wait_for_load_state("networkidle", timeout=6000)
        except Exception:
            pass
        try:
            await page.evaluate("window.scrollTo(0, Math.min(document.body.scrollHeight, 2200))")
            await page.wait_for_timeout(400)
            await page.evaluate("window.scrollTo(0, 0)")
        except Exception:
            pass

    async def _visit(self, page: Page, url: str, site_root: str) -> tuple[PageSnapshot, list[TransitionEdge], str]:
        await page.goto(url, wait_until="domcontentloaded")
        await self._prepare_page(page)
        current_url = _normalize_url(page.url)
        preview_path, preview_url = self._preview_path(site_root, current_url)
        try:
            await page.screenshot(path=str(preview_path), full_page=False)
        except Exception:
            preview_url = None
        html = await page.content()
        snapshot, transitions = _extract_page(current_url, html, preview_url)
        filtered = [edge for edge in transitions if edge.to_url and _same_site(edge.to_url, site_root)]
        return snapshot, filtered, current_url

    async def crawl(self, root_url: str, max_pages: int = 15) -> SiteKnowledgeBase:
        root_url = _normalize_url(root_url)
        domain = _base_domain(root_url)
        visited: set[str] = set()
        queue = deque([root_url])
        pages: list[PageSnapshot] = []
        transitions: list[TransitionEdge] = []

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await self._new_context(browser)
            page = await context.new_page()

            while queue and len(visited) < max_pages:
                candidate = _normalize_url(queue.popleft())
                if candidate in visited or not _same_site(candidate, root_url):
                    continue
                try:
                    snapshot, page_transitions, final_url = await self._visit(page, candidate, root_url)
                except Exception:
                    visited.add(candidate)
                    continue

                if final_url in visited:
                    continue

                visited.add(final_url)
                pages.append(snapshot)
                transitions.extend(page_transitions)

                for edge in page_transitions:
                    normalized = _normalize_url(edge.to_url)
                    if normalized not in visited and _same_site(normalized, root_url):
                        queue.append(normalized)

            await context.close()
            await browser.close()

        return SiteKnowledgeBase(root_url=root_url, domain=domain, pages=pages, transitions=transitions)
