from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable
from typing import Any, Literal
from urllib.parse import urlparse

from agent.embeddings import EmbeddingClient
from agent.llm import LLMClient
from core.models import ActionPlan, ActionStep, FieldDescriptor, FormDescriptor, PageSnapshot, SiteKnowledgeBase


logger = logging.getLogger("web_agent.agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


Intent = Literal["answer", "action", "clarification"]

QUESTION_OPENERS = (
    "как",
    "где",
    "что",
    "какие",
    "какой",
    "какая",
    "когда",
    "сколько",
    "есть ли",
    "можно ли",
    "подскажи",
    "скажи",
    "расскажи",
)
ACTION_WORDS = (
    "запиши",
    "записать",
    "оформи",
    "забронируй",
    "купи",
    "оплати",
    "закажи",
    "найди и открой",
    "помоги оформить",
    "заполн",
    "оставь заявку",
    "оставь обращение",
    "заполни форму",
)
QUERY_TAG_RULES = {
    "doctors": ("врач", "doctor", "специалист"),
    "clinics": ("клиник", "clinic", "центр", "больниц", "стационар"),
    "services": ("услуг", "service", "направлен"),
    "analyses": ("анализ", "lab"),
    "booking": ("запис", "appointment", "book", "заявк", "звонок"),
    "contacts": ("контакт", "адрес", "телефон", "phone"),
    "prices": ("цен", "стоим", "price"),
}
ACTION_ENTRY_PATTERNS = ("запис", "appointment", "book", "заявк", "звонок", "обратн", "callback", "связ")
IRREVERSIBLE_BUTTON_PATTERNS = ("отправ", "submit", "confirm", "подтверд", "оплат", "delete")
FIELD_HINTS = {
    "phone": ("phone", "тел", "моб", "номер"),
    "full_name": ("name", "фио", "имя", "пациент", "фамил"),
    "date": ("date", "дат", "день"),
    "time": ("time", "врем", "час"),
    "branch": ("branch", "clinic", "филиал", "адрес", "location"),
    "email": ("email", "e-mail", "почт"),
    "comment": ("comment", "коммент", "message", "сообщен", "пожелан"),
}
MAX_GRAPH_OVERVIEW_CHARS = 2400
MAX_PAGE_CONTEXT_CHARS = 2200
MAX_PROMPT_CONTEXT_CHARS = 12000
MAX_GRAPH_NEIGHBORS = 6

PLAN_SYSTEM_PROMPT = """
You are a website action planner. Use only the supplied site knowledge base.
Return strictly valid JSON with this schema:
{
  "goal": "string",
  "possible": true,
  "missing_inputs": ["field"],
  "steps": [
    {
      "action": "goto|click|type|select|wait_for",
      "url": "optional",
      "selector": "optional",
      "value": "optional",
      "option": "optional",
      "text": "optional",
      "reason": "optional",
      "expected_url_contains": "optional",
      "expected_selector": "optional",
      "expected_text": "optional"
    }
  ],
  "stop_before": [
    {
      "action": "click",
      "selector": "button[type=submit]"
    }
  ],
  "explanation_for_user": "string"
}
Rules:
- Use only URLs and elements from the provided site context.
- Use the site graph to pick the most relevant page or route.
- Prefer pages that have matching forms, fields and buttons.
- Never include submit/confirm/pay/delete as executable steps. Put them in stop_before.
- If data is missing, list exact missing_inputs and do not invent values.
- If the goal is not supported by the site context, set possible=false.
""".strip()

ANSWER_SYSTEM_PROMPT = """
You are a concise website assistant. Use only the supplied site knowledge base snippets.
Return strictly valid JSON:
{
  "reply": "short answer in Russian",
  "suggested_url": "optional string",
  "relevant_pages": ["url"],
  "missing_inputs": ["field"]
}
Rules:
- Answer briefly and directly. Do not dump the raw site context.
- Use the site graph and selected pages to choose the best page to open in the background.
- Mention only the most relevant 1-3 sections/pages.
- If the user asked a question, do not output action steps.
- If the answer is uncertain, say that clearly.
""".strip()


def _normalize_site(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return parsed._replace(netloc=netloc, fragment="").geturl()


def _same_site(a: str, b: str) -> bool:
    a_host = urlparse(_normalize_site(a)).netloc
    b_host = urlparse(_normalize_site(b)).netloc
    return a_host == b_host or a_host.endswith(f".{b_host}") or b_host.endswith(f".{a_host}")


def _clean_text(value: str | None) -> str:
    return (value or "").strip().lower()


class AgentService:
    def __init__(self) -> None:
        self.llm = LLMClient()
        self.embeddings = EmbeddingClient()
        self._memory: dict[str, dict[str, Any]] = {}
        self._query_embedding_cache: dict[str, list[float]] = {}

    def _memory_key(self, site_url: str) -> str:
        return _normalize_site(site_url)

    def _remember(self, site_url: str, *, last_goal: str | None, last_mode: str, missing_inputs: Iterable[str]) -> None:
        self._memory[self._memory_key(site_url)] = {
            "last_goal": last_goal,
            "last_mode": last_mode,
            "missing_inputs": list(missing_inputs),
        }

    def _recall(self, site_url: str) -> dict[str, Any]:
        return self._memory.get(self._memory_key(site_url), {})

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in re.split(r"\W+", text.lower()) if len(token) > 2]

    def _query_tags(self, query: str) -> set[str]:
        lowered = query.lower()
        return {tag for tag, patterns in QUERY_TAG_RULES.items() if any(pattern in lowered for pattern in patterns)}

    def _page_embedding_text(self, page: PageSnapshot) -> str:
        chunks: list[str] = []
        if page.title:
            chunks.append(page.title)
        if page.summary:
            chunks.append(page.summary)
        if page.tags:
            chunks.append(" ".join(page.tags))
        if page.headings:
            chunks.append(" ".join(page.headings[:12]))
        if page.text_content:
            chunks.append(page.text_content[:5000])
        if page.visible_text_sample:
            chunks.append(" ".join(page.visible_text_sample[:12]))
        link_texts = [link.text for link in page.links[:30] if link.text]
        if link_texts:
            chunks.append(" ".join(link_texts))
        button_texts = [button.text for button in page.buttons[:20] if button.text]
        if button_texts:
            chunks.append(" ".join(button_texts))
        field_texts = [field.label or field.name or field.placeholder or "" for field in page.inputs[:20]]
        if field_texts:
            chunks.append(" ".join(text for text in field_texts if text))
        return "\n".join(chunk for chunk in chunks if chunk).strip()[:6000]

    async def enrich_kb_embeddings(self, kb: SiteKnowledgeBase) -> SiteKnowledgeBase:
        if not self.embeddings.enabled:
            logger.warning("Embeddings are disabled; KB will use lexical search only. site=%s", kb.root_url)
            return kb

        pages_to_embed = [page for page in kb.pages if not page.embedding and self._page_embedding_text(page)]
        if not pages_to_embed:
            return kb

        page_texts = [self._page_embedding_text(page) for page in pages_to_embed]
        try:
            vectors = await self.embeddings.embed_texts(page_texts, request_name="kb_page_index")
        except Exception:
            logger.exception("Failed to build page embeddings for site=%s", kb.root_url)
            return kb

        vector_by_url = {page.url: vector for page, vector in zip(pages_to_embed, vectors, strict=False)}
        kb.pages = [page.model_copy(update={"embedding": vector_by_url.get(page.url, page.embedding)}) for page in kb.pages]
        logger.info("Embedded %s pages for site=%s", len(vector_by_url), kb.root_url)
        return kb

    async def _query_embedding(self, query: str) -> list[float] | None:
        if not self.embeddings.enabled:
            return None
        cache_key = query.strip().lower()
        if not cache_key:
            return None
        cached = self._query_embedding_cache.get(cache_key)
        if cached:
            return cached
        try:
            vectors = await self.embeddings.embed_texts([query], request_name="query_search")
        except Exception:
            logger.exception("Failed to build query embedding for query=%s", query)
            return None
        vector = vectors[0] if vectors else None
        if vector:
            self._query_embedding_cache[cache_key] = vector
        return vector

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _extract_inputs(self, message: str, user_inputs: dict[str, Any]) -> dict[str, Any]:
        merged = {key: value for key, value in user_inputs.items() if value not in ("", None)}
        lowered = message.lower()

        phone_match = re.search(r"(\+?\d[\d\-\s\(\)]{8,}\d)", message)
        if phone_match:
            merged["phone"] = re.sub(r"\s+", "", phone_match.group(1))

        date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", message)
        if date_match:
            merged["date"] = date_match.group(1)

        time_match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", message)
        if time_match:
            merged["time"] = f"{time_match.group(1).zfill(2)}:{time_match.group(2)}"

        name_match = re.search(r"(?:меня зовут|фио|имя)\s*[:\-]?\s*([А-ЯA-ZЁ][А-ЯA-ZЁа-яa-zё\-]+\s+[А-ЯA-ZЁ][А-ЯA-ZЁа-яa-zё\-]+(?:\s+[А-ЯA-ZЁ][А-ЯA-ZЁа-яa-zё\-]+)?)", message)
        if name_match:
            merged["full_name"] = name_match.group(1).strip()

        branch_match = re.search(r"(?:филиал|клиника|адрес)\s*[:\-]?\s*([^\n,]{3,80})", lowered)
        if branch_match:
            merged["branch"] = branch_match.group(1).strip()

        return merged

    def _classify_intent(self, site_url: str, message: str, collected_inputs: dict[str, Any]) -> Intent:
        lowered = _clean_text(message)
        recall = self._recall(site_url)

        if lowered in {"выполни план", "выполни", "запусти", "продолжай", "продолжи"} and recall.get("last_goal"):
            return "clarification"

        if any(word in lowered for word in ACTION_WORDS):
            return "action"

        if lowered.endswith("?") or lowered.startswith(QUESTION_OPENERS):
            return "answer"

        if recall.get("last_mode") == "action" and collected_inputs and recall.get("last_goal"):
            return "clarification"

        if any(word in lowered for word in ("скажи", "подскажи", "какие", "какой", "есть ли", "можно ли")):
            return "answer"

        return "action"

    def _matching_snippets(self, page: PageSnapshot, query: str, limit: int = 6) -> list[str]:
        terms = self._tokenize(query)
        pool: list[str] = []
        pool.extend(page.headings)
        pool.extend(page.visible_text_sample)
        pool.extend(link.text for link in page.links[:30] if link.text)
        pool.extend(button.text for button in page.buttons[:20] if button.text)
        pool.extend(field.label or field.name or "" for field in page.inputs[:20])
        if page.text_excerpt:
            pool.extend(re.split(r"(?<=[.!?])\s+", page.text_excerpt))

        scored: list[tuple[int, str]] = []
        seen: set[str] = set()
        for raw in pool:
            snippet = raw.strip()
            if not snippet or snippet in seen or len(snippet) < 3:
                continue
            seen.add(snippet)
            lowered = snippet.lower()
            score = sum(lowered.count(term) for term in terms)
            if score:
                scored.append((score, snippet[:240]))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [snippet for _, snippet in scored[:limit]]

    def _extract_person_names(self, page: PageSnapshot, limit: int = 25) -> list[str]:
        name_re = re.compile(r"^[А-ЯЁ][а-яё-]+\s+[А-ЯЁ][а-яё-]+(?:\s+[А-ЯЁ][а-яё-]+)?$")
        names: list[str] = []
        seen: set[str] = set()

        for link in page.links[:300]:
            text = (link.text or "").strip()
            if text not in seen and name_re.match(text):
                seen.add(text)
                names.append(text)

        for snippet in page.visible_text_sample:
            text = snippet.strip()
            if text not in seen and name_re.match(text):
                seen.add(text)
                names.append(text)

        return names[:limit]

    def _extract_doctor_profiles(self, page: PageSnapshot, limit: int = 300) -> list[dict[str, str]]:
        name_re = re.compile(r"^[А-ЯЁ][а-яё-]+\s+[А-ЯЁ][а-яё-]+(?:\s+[А-ЯЁ][а-яё-]+)?$")
        profiles: list[dict[str, str]] = []
        seen: set[str] = set()
        links = page.links

        for index, link in enumerate(links):
            name = (link.text or "").strip()
            href = link.href or ""
            if not name_re.match(name):
                continue
            if name in seen:
                continue
            seen.add(name)

            specialty = ""
            for offset in (1, -1, 2):
                neighbor_index = index + offset
                if 0 <= neighbor_index < len(links):
                    neighbor = links[neighbor_index]
                    neighbor_text = (neighbor.text or "").strip()
                    neighbor_href = neighbor.href or ""
                    if "/Departments/" in neighbor_href or "/Poly/Sections/" in neighbor_href:
                        specialty = neighbor_text
                        break

            profiles.append({"name": name, "specialty": specialty, "url": href})

        return profiles[:limit]

    def _specific_doctor_result(self, kb: SiteKnowledgeBase, question: str) -> dict[str, Any] | None:
        lowered = question.lower()
        if not any(word in lowered for word in ("врач", "доктор", "специалист")):
            return None

        name_re = re.compile(r"[А-ЯЁ][а-яё-]+\s+[А-ЯЁ][а-яё-]+(?:\s+[А-ЯЁ][а-яё-]+)?")
        match = name_re.search(question)
        if not match:
            return None

        target_name = match.group(0).strip().lower()
        for page in kb.pages:
            for profile in self._extract_doctor_profiles(page, limit=500):
                if profile["name"].lower() == target_name:
                    specialty = f" Специализация или отделение: {profile['specialty']}." if profile["specialty"] else ""
                    return {
                        "reply": f"Нашёл врача {profile['name']}.{specialty} Открываю его страницу.",
                        "suggested_url": profile["url"] or page.url,
                        "relevant_pages": [profile["url"] or page.url],
                        "missing_inputs": [],
                    }
        return None

    def _doctor_list_answer(self, kb: SiteKnowledgeBase, question: str, ranked_pages: list[PageSnapshot] | None = None) -> dict[str, Any] | None:
        lowered = question.lower()
        if not any(token in lowered for token in ("какие врачи", "список врач", "врачи", "специалист")):
            return None

        source_pages = ranked_pages or self._rank_pages(kb, question, action_mode=False, limit=5)
        doctor_pages = [page for page in source_pages if "doctors" in page.tags or "employees" in page.url.lower()]
        if not doctor_pages:
            return None

        profiles: list[dict[str, str]] = []
        seen: set[str] = set()
        for page in doctor_pages[:3]:
            for profile in self._extract_doctor_profiles(page, limit=500):
                if profile["name"] not in seen:
                    seen.add(profile["name"])
                    profiles.append(profile)
        if not profiles:
            return None

        doctor_items = []
        for profile in profiles:
            if profile["specialty"]:
                doctor_items.append(f"{profile['name']} — {profile['specialty']}")
            else:
                doctor_items.append(profile["name"])
        reply = "На сайте нашёл раздел со специалистами. Список врачей: " + "; ".join(doctor_items) + "."
        reply += " Если хотите, назовите конкретного врача, и я открою его страницу в превью."
        return {
            "reply": reply,
            "suggested_url": doctor_pages[0].url,
            "relevant_pages": [page.url for page in doctor_pages[:3]],
            "missing_inputs": [],
        }

    def _is_navigation_request(self, message: str) -> bool:
        lowered = message.lower()
        return any(token in lowered for token in ("открой", "перейди", "покажи", "найди раздел"))

    def _page_lookup(self, kb: SiteKnowledgeBase) -> dict[str, PageSnapshot]:
        return {_normalize_site(page.url): page for page in kb.pages}

    def _graph_adjacency(self, kb: SiteKnowledgeBase) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        outgoing: dict[str, list[str]] = {}
        incoming: dict[str, list[str]] = {}
        page_lookup = self._page_lookup(kb)

        for edge in kb.transitions:
            from_url = _normalize_site(edge.from_url)
            to_url = _normalize_site(edge.to_url)
            if from_url not in page_lookup or to_url not in page_lookup:
                continue
            outgoing.setdefault(from_url, [])
            if to_url not in outgoing[from_url]:
                outgoing[from_url].append(to_url)
            incoming.setdefault(to_url, [])
            if from_url not in incoming[to_url]:
                incoming[to_url].append(from_url)

        return outgoing, incoming

    def _graph_hubs(self, kb: SiteKnowledgeBase, limit: int = 6) -> list[PageSnapshot]:
        page_lookup = self._page_lookup(kb)
        outgoing, incoming = self._graph_adjacency(kb)
        scored: list[tuple[int, PageSnapshot]] = []
        for page in kb.pages:
            normalized = _normalize_site(page.url)
            degree = len(outgoing.get(normalized, [])) + len(incoming.get(normalized, []))
            scored.append((degree, page))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [page for degree, page in scored[:limit] if degree > 0 and _normalize_site(page.url) in page_lookup]

    def _short_page_label(self, page: PageSnapshot) -> str:
        return page.title or page.summary or page.url

    def _graph_neighbors(self, kb: SiteKnowledgeBase, pages: list[PageSnapshot], limit: int = MAX_GRAPH_NEIGHBORS) -> list[PageSnapshot]:
        page_lookup = self._page_lookup(kb)
        outgoing, incoming = self._graph_adjacency(kb)
        neighbor_urls: list[str] = []
        seen: set[str] = {_normalize_site(page.url) for page in pages}

        for page in pages:
            normalized = _normalize_site(page.url)
            candidates = outgoing.get(normalized, [])[:4] + incoming.get(normalized, [])[:2]
            for url in candidates:
                if url in seen:
                    continue
                seen.add(url)
                neighbor_urls.append(url)
                if len(neighbor_urls) >= limit:
                    break
            if len(neighbor_urls) >= limit:
                break

        return [page_lookup[url] for url in neighbor_urls if url in page_lookup]

    def _graph_path(self, kb: SiteKnowledgeBase, from_url: str, to_url: str, max_depth: int = 4) -> list[str]:
        start = _normalize_site(from_url)
        goal = _normalize_site(to_url)
        if start == goal:
            return [start]

        outgoing, _ = self._graph_adjacency(kb)
        queue: list[tuple[str, list[str]]] = [(start, [start])]
        seen = {start}
        while queue:
            current, path = queue.pop(0)
            if len(path) > max_depth:
                continue
            for neighbor in outgoing.get(current, []):
                if neighbor in seen:
                    continue
                next_path = path + [neighbor]
                if neighbor == goal:
                    return next_path
                seen.add(neighbor)
                queue.append((neighbor, next_path))
        return []

    def _graph_overview(self, kb: SiteKnowledgeBase, selected_pages: list[PageSnapshot]) -> str:
        root = str(kb.root_url)
        page_lookup = self._page_lookup(kb)
        outgoing, incoming = self._graph_adjacency(kb)
        lines = [f"SITE_ROOT: {root}", f"PAGES: {len(kb.pages)}", f"TRANSITIONS: {len(kb.transitions)}"]

        hubs = self._graph_hubs(kb, limit=5)
        if hubs:
            lines.append("GRAPH_HUBS:")
            for page in hubs:
                normalized = _normalize_site(page.url)
                lines.append(
                    f"- {self._short_page_label(page)} | url={page.url} | out={len(outgoing.get(normalized, []))} | in={len(incoming.get(normalized, []))}"
                )

        if selected_pages:
            lines.append("SELECTED_PATHS:")
            for page in selected_pages[:4]:
                path = self._graph_path(kb, root, page.url)
                if not path:
                    lines.append(f"- {page.url}")
                    continue
                path_labels = [self._short_page_label(page_lookup[node]) for node in path if node in page_lookup]
                lines.append(f"- {' -> '.join(path_labels[:5])}")

        overview = "\n".join(lines)
        return overview[:MAX_GRAPH_OVERVIEW_CHARS]

    def _page_context_block(self, page: PageSnapshot, *, action_mode: bool, query: str) -> str:
        lines = [f"PAGE: {page.url}"]
        if page.title:
            lines.append(f"TITLE: {page.title}")
        if page.summary:
            lines.append(f"SUMMARY: {page.summary}")
        if page.tags:
            lines.append(f"TAGS: {', '.join(page.tags)}")
        if page.headings:
            lines.append(f"HEADINGS: {', '.join(page.headings[:6])}")
        if action_mode:
            if page.forms:
                for form in page.forms[:3]:
                    fields = [field.label or field.name or field.selector for field in form.fields[:10]]
                    buttons = [button.text or button.selector for button in form.buttons[:6]]
                    lines.append(f"FORM: selector={form.selector} action={form.action} fields={fields} buttons={buttons}")
            elif page.inputs:
                fields = [field.label or field.name or field.selector for field in page.inputs[:10]]
                lines.append(f"INPUTS: {fields}")
            if page.buttons:
                button_lines = []
                for button in page.buttons[:10]:
                    target = f" -> {button.target_url}" if button.target_url else ""
                    button_lines.append(f"{button.text or button.selector}{target}")
                lines.append(f"BUTTONS: {' | '.join(button_lines)}")
        else:
            snippets = self._matching_snippets(page, query, limit=6)
            if snippets:
                lines.append(f"SNIPPETS: {' | '.join(snippets)}")
            elif page.visible_text_sample:
                lines.append(f"TEXT: {' | '.join(page.visible_text_sample[:5])}")
        block = "\n".join(lines)
        return block[:MAX_PAGE_CONTEXT_CHARS]

    async def _retrieve_context_bundle(
        self,
        kb: SiteKnowledgeBase,
        query: str,
        *,
        action_mode: bool,
        primary_limit: int,
        llm_limit: int,
    ) -> tuple[list[PageSnapshot], list[PageSnapshot], str]:
        primary_pages = await self._rank_pages_semantic(kb, query, action_mode=action_mode, limit=primary_limit)
        neighbor_pages = self._graph_neighbors(kb, primary_pages, limit=MAX_GRAPH_NEIGHBORS)
        llm_pages: list[PageSnapshot] = []
        seen: set[str] = set()
        for page in [*primary_pages, *neighbor_pages]:
            normalized = _normalize_site(page.url)
            if normalized in seen:
                continue
            seen.add(normalized)
            llm_pages.append(page)
            if len(llm_pages) >= llm_limit:
                break

        graph_overview = self._graph_overview(kb, llm_pages)
        blocks: list[str] = []
        total_chars = len(graph_overview)
        for page in llm_pages:
            block = self._page_context_block(page, action_mode=action_mode, query=query)
            if total_chars + len(block) > MAX_PROMPT_CONTEXT_CHARS:
                break
            blocks.append(block)
            total_chars += len(block)

        context = graph_overview
        if blocks:
            context += "\n\nPAGE_CONTEXT:\n" + "\n\n".join(blocks)
        return primary_pages, llm_pages, context

    def _field_text(self, field: FieldDescriptor) -> str:
        return " ".join(
            value for value in [field.name or "", field.label or "", field.placeholder or "", field.input_type or ""] if value
        ).lower()

    def _field_semantic(self, field: FieldDescriptor) -> str | None:
        haystack = self._field_text(field)
        for semantic, patterns in FIELD_HINTS.items():
            if any(pattern in haystack for pattern in patterns):
                return semantic
        return None

    def _goal_action_kind(self, goal: str) -> str:
        lowered = goal.lower()
        if self._is_navigation_request(goal):
            return "navigation"
        if any(token in lowered for token in ("запис", "прием", "приём", "appointment", "book")):
            return "booking"
        if any(token in lowered for token in ("обратный звонок", "перезвон", "callback", "свяж")):
            return "callback"
        if any(token in lowered for token in ("заявк", "обращен", "форм", "оставь")):
            return "lead"
        return "generic"

    def _target_score(self, page: PageSnapshot, goal: str) -> int:
        score = self._page_score(page, goal, action_mode=True)
        if page.forms:
            score += 20
        if page.inputs:
            score += 8
        if any(any(pattern in (button.text or "").lower() for pattern in ACTION_ENTRY_PATTERNS) for button in page.buttons[:20]):
            score += 6
        return score

    def _entry_target_urls(self, page: PageSnapshot, goal: str) -> list[str]:
        goal_tags = self._query_tags(goal)
        urls: list[str] = []
        for button in page.buttons:
            text = (button.text or "").lower()
            if button.target_url and (any(pattern in text for pattern in ACTION_ENTRY_PATTERNS) or goal_tags & {"booking", "doctors"}):
                urls.append(button.target_url)
        for link in page.links[:80]:
            text = (link.text or "").lower()
            if link.href and (any(pattern in text for pattern in ACTION_ENTRY_PATTERNS) or goal_tags & {"booking", "doctors"}):
                urls.append(link.href)
        deduped: list[str] = []
        seen: set[str] = set()
        for url in urls:
            normalized = _normalize_site(url)
            if normalized not in seen:
                seen.add(normalized)
                deduped.append(url)
        return deduped

    def _choose_form(self, page: PageSnapshot, goal: str) -> FormDescriptor | None:
        if not page.forms:
            return None
        kind = self._goal_action_kind(goal)
        scored: list[tuple[int, FormDescriptor]] = []
        for form in page.forms:
            score = len(form.fields) * 2 + len(form.buttons)
            texts = " ".join(
                [button.text or "" for button in form.buttons] + [field.label or field.name or "" for field in form.fields]
            ).lower()
            if kind == "booking" and any(token in texts for token in ("запис", "прием", "приём", "appointment")):
                score += 10
            if kind == "callback" and any(token in texts for token in ("звон", "callback", "связ")):
                score += 10
            if any(field.required for field in form.fields):
                score += 4
            scored.append((score, form))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1] if scored else None

    def _find_action_target(
        self,
        kb: SiteKnowledgeBase,
        goal: str,
        ranked_pages: list[PageSnapshot] | None = None,
    ) -> tuple[PageSnapshot | None, PageSnapshot | None]:
        ranked_pages = ranked_pages or self._rank_pages(kb, goal, action_mode=True, limit=8)
        if not ranked_pages:
            return None, None

        page_lookup = self._page_lookup(kb)
        best_entry = ranked_pages[0]
        best_target = best_entry

        direct_candidates = sorted(ranked_pages, key=lambda page: self._target_score(page, goal), reverse=True)
        for page in direct_candidates:
            if page.forms or page.inputs:
                return best_entry, page

        for entry_page in direct_candidates:
            for target_url in self._entry_target_urls(entry_page, goal):
                target = page_lookup.get(_normalize_site(target_url))
                if target and (target.forms or target.inputs or "booking" in target.tags or "lead_form" in target.tags):
                    return entry_page, target

        return best_entry, best_target

    def _required_semantics(self, goal: str, target_page: PageSnapshot, form: FormDescriptor | None) -> set[str]:
        required: set[str] = set()
        kind = self._goal_action_kind(goal)
        if kind == "booking":
            required.update({"full_name", "phone"})
        if kind in {"callback", "lead"}:
            required.add("phone")
        fields = form.fields if form else target_page.inputs
        for field in fields:
            semantic = self._field_semantic(field)
            if semantic and field.required:
                required.add(semantic)
        return required

    def _form_fields(self, page: PageSnapshot, form: FormDescriptor | None) -> list[FieldDescriptor]:
        return form.fields if form and form.fields else page.inputs

    def _stop_before_steps(self, page: PageSnapshot, form: FormDescriptor | None) -> list[ActionStep]:
        buttons = form.buttons if form and form.buttons else page.buttons
        for button in buttons:
            if button.selector and any(term in _clean_text(button.text) for term in IRREVERSIBLE_BUTTON_PATTERNS):
                return [ActionStep(action="click", selector=button.selector, reason="Финальное действие должен выполнить человек.")]
        return []

    def _deterministic_plan(
        self,
        kb: SiteKnowledgeBase,
        goal: str,
        user_inputs: dict[str, Any],
        ranked_pages: list[PageSnapshot] | None = None,
    ) -> tuple[ActionPlan, list[PageSnapshot]]:
        ranked_pages = ranked_pages or self._rank_pages(kb, goal, action_mode=True, limit=8)
        if self._is_navigation_request(goal):
            target_page = ranked_pages[0] if ranked_pages else None
            if not target_page:
                return (
                    ActionPlan(
                        goal=goal,
                        possible=False,
                        missing_inputs=[],
                        steps=[],
                        stop_before=[],
                        explanation_for_user="Не нашёл релевантный раздел сайта для перехода.",
                    ),
                    [],
                )
            plan = ActionPlan(
                goal=goal,
                possible=True,
                missing_inputs=[],
                steps=[
                    ActionStep(
                        action="goto",
                        url=target_page.url,
                        reason="Открыть релевантный раздел сайта.",
                        expected_url_contains=urlparse(target_page.url).path or "/",
                    )
                ],
                stop_before=[],
                explanation_for_user="Открываю нужный раздел сайта.",
            )
            return self._sanitize_plan(kb, plan, target_page.url), [target_page]

        entry_page, target_page = self._find_action_target(kb, goal, ranked_pages=ranked_pages)
        if not target_page:
            return (
                ActionPlan(
                    goal=goal,
                    possible=False,
                    missing_inputs=[],
                    steps=[],
                    stop_before=[],
                    explanation_for_user="Не нашёл на сайте подходящую страницу или форму для этой задачи.",
                ),
                [],
            )

        form = self._choose_form(target_page, goal)
        fields = self._form_fields(target_page, form)
        if not fields and not form:
            relevant_pages = [page for page in [entry_page, target_page] if page]
            return (
                ActionPlan(
                    goal=goal,
                    possible=False,
                    missing_inputs=[],
                    steps=[],
                    stop_before=[],
                    explanation_for_user="Нашёл релевантную страницу, но не нашёл на ней форму или поля для безопасного заполнения.",
                ),
                relevant_pages,
            )
        required_semantics = self._required_semantics(goal, target_page, form)
        missing_inputs = self._missing_inputs(goal, user_inputs)

        steps: list[ActionStep] = [
            ActionStep(
                action="goto",
                url=target_page.url,
                reason="Открыть целевую страницу с формой или нужным интерфейсом.",
                expected_url_contains=urlparse(target_page.url).path or "/",
            )
        ]

        if form and form.selector:
            steps.append(ActionStep(action="wait_for", selector=form.selector, reason="Дождаться формы на странице."))
        elif fields and fields[0].selector:
            steps.append(ActionStep(action="wait_for", selector=fields[0].selector, reason="Дождаться первого поля формы."))

        used_semantics: set[str] = set()
        fill_step_count = 0
        for field in fields[:16]:
            selector = field.selector
            semantic = self._field_semantic(field)
            if not selector or not semantic or semantic in used_semantics:
                continue
            used_semantics.add(semantic)

            raw_value = user_inputs.get(semantic)
            if semantic in required_semantics and raw_value in ("", None):
                if semantic not in missing_inputs:
                    missing_inputs.append(semantic)
                continue

            if raw_value in ("", None):
                continue

            if field.options or (field.input_type or "").lower() == "select":
                steps.append(
                    ActionStep(
                        action="select",
                        selector=selector,
                        option=f"{{{{{semantic}}}}}",
                        reason=f"Заполнить поле {semantic}.",
                        expected_selector=selector,
                    )
                )
                fill_step_count += 1
            else:
                steps.append(
                    ActionStep(
                        action="type",
                        selector=selector,
                        value=f"{{{{{semantic}}}}}",
                        reason=f"Заполнить поле {semantic}.",
                        expected_selector=selector,
                    )
                )
                fill_step_count += 1

        if self._goal_action_kind(goal) != "navigation" and fill_step_count == 0:
            relevant_pages = [page for page in [entry_page, target_page] if page]
            return (
                ActionPlan(
                    goal=goal,
                    possible=False,
                    missing_inputs=missing_inputs,
                    steps=[],
                    stop_before=[],
                    explanation_for_user="Нашёл страницу, но не смог надёжно сопоставить поля формы с данными пользователя. Для этого сайта нужен более точный сценарий.",
                ),
                relevant_pages,
            )

        stop_before = self._stop_before_steps(target_page, form)
        explanation = "Построен детерминированный план по карте сайта."
        if missing_inputs:
            explanation += f" Не хватает данных: {', '.join(missing_inputs)}."
        elif stop_before:
            explanation += " Поля будут заполнены автоматически, затем агент остановится перед финальным подтверждением."
        else:
            explanation += " На странице не найдено явной необратимой кнопки, поэтому выполню только безопасные шаги."

        plan = ActionPlan(
            goal=goal,
            possible=True,
            missing_inputs=missing_inputs,
            steps=steps,
            stop_before=stop_before,
            explanation_for_user=explanation,
        )
        relevant_pages = [page for page in [entry_page, target_page] if page]
        deduped_pages: list[PageSnapshot] = []
        seen_urls: set[str] = set()
        for page in relevant_pages:
            if page.url not in seen_urls:
                seen_urls.add(page.url)
                deduped_pages.append(page)
        return self._sanitize_plan(kb, plan, target_page.url), deduped_pages

    def _page_texts(self, page: PageSnapshot) -> dict[str, str]:
        return {
            "url": page.url.lower(),
            "title": (page.title or "").lower(),
            "tags": " ".join(page.tags).lower(),
            "headings": " ".join(page.headings).lower(),
            "buttons": " ".join(button.text or "" for button in page.buttons).lower(),
            "links": " ".join(link.text or "" for link in page.links).lower(),
            "fields": " ".join((field.label or field.name or "") for field in page.inputs).lower(),
            "text": (page.text_excerpt or "").lower(),
        }

    def _page_score(self, page: PageSnapshot, query: str, *, action_mode: bool) -> int:
        terms = self._tokenize(query)
        qtags = self._query_tags(query)
        texts = self._page_texts(page)
        score = 0

        for term in terms:
            if term in texts["url"]:
                score += 10
            if term in texts["title"]:
                score += 8
            if term in texts["headings"]:
                score += 6
            if term in texts["buttons"]:
                score += 4
            if term in texts["links"]:
                score += 4
            if term in texts["fields"]:
                score += 3
            score += min(texts["text"].count(term), 6)

        for tag in qtags:
            if tag in page.tags:
                score += 12

        if action_mode:
            if "has_form" in page.tags:
                score += 10
            if "booking" in qtags and "booking" in page.tags:
                score += 12
            if "booking" in qtags and "lead_form" in page.tags:
                score += 6
            if "doctors" in qtags and "doctors" in page.tags:
                score += 8
            if "children" in page.tags and "реб" not in query.lower() and "дет" not in query.lower():
                score -= 8
            if "beauty" in page.tags and "космет" not in query.lower():
                score -= 8
        else:
            if "faq" in page.tags:
                score += 4
            if qtags & set(page.tags):
                score += 6

        return score

    def _rank_pages(self, kb: SiteKnowledgeBase, query: str, *, action_mode: bool, limit: int = 5) -> list[PageSnapshot]:
        scored = [(self._page_score(page, query, action_mode=action_mode), page) for page in kb.pages]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [page for score, page in scored[:limit] if score > 0]

    async def _rank_pages_semantic(self, kb: SiteKnowledgeBase, query: str, *, action_mode: bool, limit: int = 5) -> list[PageSnapshot]:
        lexical_scores = {page.url: self._page_score(page, query, action_mode=action_mode) for page in kb.pages}
        query_vector = await self._query_embedding(query)
        if not query_vector:
            return self._rank_pages(kb, query, action_mode=action_mode, limit=limit)

        scored: list[tuple[float, PageSnapshot]] = []
        for page in kb.pages:
            semantic_score = self._cosine_similarity(query_vector, page.embedding)
            lexical_score = float(lexical_scores.get(page.url, 0))
            combined_score = lexical_score + (semantic_score * 30.0)
            if semantic_score > 0:
                combined_score += 2.0
            scored.append((combined_score, page))

        scored.sort(key=lambda item: item[0], reverse=True)
        ranked_pages = [page for score, page in scored if score > 0][:limit]
        if ranked_pages:
            return ranked_pages
        return self._rank_pages(kb, query, action_mode=action_mode, limit=limit)

    def _answer_context(self, page: PageSnapshot, query: str) -> str:
        snippets = self._matching_snippets(page, query, limit=6)
        lines = [f"PAGE: {page.url}"]
        if page.title:
            lines.append(f"TITLE: {page.title}")
        if page.tags:
            lines.append(f"TAGS: {', '.join(page.tags)}")
        if page.headings:
            lines.append(f"HEADINGS: {', '.join(page.headings[:5])}")
        if snippets:
            lines.append(f"SNIPPETS: {' | '.join(snippets)}")
        return "\n".join(lines)

    def _action_context(self, page: PageSnapshot) -> str:
        lines = [f"PAGE: {page.url}"]
        if page.title:
            lines.append(f"TITLE: {page.title}")
        if page.tags:
            lines.append(f"TAGS: {', '.join(page.tags)}")
        if page.headings:
            lines.append(f"HEADINGS: {', '.join(page.headings[:6])}")
        if page.forms:
            for form in page.forms[:4]:
                fields = [field.label or field.name or field.selector for field in form.fields[:10]]
                buttons = [button.text or button.selector for button in form.buttons[:6]]
                lines.append(f"FORM: action={form.action} fields={fields} buttons={buttons}")
        if page.buttons:
            button_lines = []
            for button in page.buttons[:12]:
                target = f" -> {button.target_url}" if button.target_url else ""
                button_lines.append(f"{button.text or button.selector}{target}")
            if button_lines:
                lines.append(f"BUTTONS: {' | '.join(button_lines)}")
        return "\n".join(lines)

    def _relevant_urls(self, kb: SiteKnowledgeBase) -> set[str]:
        urls = {page.url for page in kb.pages}
        for page in kb.pages:
            urls.update(link.href for link in page.links if link.href)
            urls.update(button.target_url for button in page.buttons if button.target_url)
            urls.update(form.action for form in page.forms if form.action)
        return {url for url in urls if url}

    def _sanitize_plan(self, kb: SiteKnowledgeBase, plan: ActionPlan, fallback_url: str) -> ActionPlan:
        known_urls = self._relevant_urls(kb)
        cleaned_steps: list[ActionStep] = []
        cleaned_stop: list[ActionStep] = []

        for step in plan.steps:
            if step.action == "goto":
                if not step.url or not _same_site(step.url, str(kb.root_url)):
                    continue
                url = step.url if step.url in known_urls else fallback_url
                cleaned_steps.append(step.model_copy(update={"url": url}))
                continue

            if step.action in {"click", "type", "select"} and not step.selector:
                continue
            cleaned_steps.append(step)

        if cleaned_steps and cleaned_steps[0].action != "goto":
            cleaned_steps.insert(0, ActionStep(action="goto", url=fallback_url, reason="Открыть релевантную страницу сайта."))
        elif not cleaned_steps and plan.possible:
            cleaned_steps.append(ActionStep(action="goto", url=fallback_url, reason="Открыть релевантную страницу сайта."))

        for step in plan.stop_before:
            if step.action == "click" and step.selector:
                cleaned_stop.append(step)

        return plan.model_copy(update={"steps": cleaned_steps, "stop_before": cleaned_stop})

    def _missing_inputs(self, goal: str, user_inputs: dict[str, Any]) -> list[str]:
        normalized = {key: value for key, value in user_inputs.items() if value not in ("", None)}
        goal_l = goal.lower()
        missing: list[str] = []

        field_rules = {
            "phone": ("телефон", "phone", "номер"),
            "full_name": ("фио", "имя", "name"),
            "date": ("дата", "date", "завтра", "сегодня"),
            "time": ("время", "time", "утро", "вечер"),
            "branch": ("филиал", "clinic", "адрес", "location"),
            "email": ("email", "почт"),
        }
        for field_name, patterns in field_rules.items():
            if any(pattern in goal_l for pattern in patterns) and field_name not in normalized:
                missing.append(field_name)

        if any(word in goal_l for word in ("запис", "appointment", "book", "прием", "приём")):
            for field_name in ("full_name", "phone"):
                if field_name not in normalized and field_name not in missing:
                    missing.append(field_name)
        return missing

    def _fallback_answer(self, kb: SiteKnowledgeBase, question: str, ranked_pages: list[PageSnapshot] | None = None) -> dict[str, Any]:
        doctor_answer = self._doctor_list_answer(kb, question, ranked_pages=ranked_pages)
        if doctor_answer:
            return doctor_answer

        relevant_pages = ranked_pages or self._rank_pages(kb, question, action_mode=False, limit=3)
        if not relevant_pages:
            return {
                "reply": "По собранной карте сайта я пока не нашёл уверенный ответ на этот вопрос.",
                "suggested_url": str(kb.root_url),
                "relevant_pages": [],
                "missing_inputs": [],
            }

        top = relevant_pages[0]
        snippets = self._matching_snippets(top, question, limit=3)
        reply_parts = []
        if top.title:
            reply_parts.append(f"На сайте есть релевантный раздел: {top.title}.")
        if snippets:
            reply_parts.append("Коротко по найденному: " + " ".join(snippets[:2]))
        else:
            reply_parts.append("Откройте этот раздел, там, вероятно, находится нужная информация.")
        reply_parts.append("Если хотите, могу открыть нужный раздел в фоне или помочь выполнить действие.")

        return {
            "reply": " ".join(reply_parts),
            "suggested_url": top.url,
            "relevant_pages": [page.url for page in relevant_pages],
            "missing_inputs": [],
        }

    def _fallback_plan(
        self,
        kb: SiteKnowledgeBase,
        goal: str,
        user_inputs: dict[str, Any],
        ranked_pages: list[PageSnapshot] | None = None,
    ) -> tuple[ActionPlan, list[PageSnapshot]]:
        return self._deterministic_plan(kb, goal, user_inputs, ranked_pages=ranked_pages)

    async def _answer_with_llm(self, kb: SiteKnowledgeBase, question: str, user_inputs: dict[str, Any]) -> dict[str, Any]:
        primary_pages, llm_pages, context = await self._retrieve_context_bundle(
            kb,
            question,
            action_mode=False,
            primary_limit=6,
            llm_limit=5,
        )
        doctor_answer = self._doctor_list_answer(kb, question, ranked_pages=primary_pages)
        if doctor_answer:
            logger.info("Doctor list answer handled without LLM for site=%s question=%s", kb.root_url, question)
            return doctor_answer

        if not llm_pages:
            logger.warning("No relevant pages found for answer mode, using fallback. site=%s question=%s", kb.root_url, question)
            return self._fallback_answer(kb, question, ranked_pages=primary_pages)

        user_prompt = (
            f"Question: {question}\n"
            f"Known user inputs: {user_inputs}\n"
            f"Relevant site graph and page context:\n{context}"
        )
        try:
            payload = await self.llm.complete_json(ANSWER_SYSTEM_PROMPT, user_prompt, request_name="answer_question")
        except Exception:
            logger.exception("LLM answer failed, using fallback. site=%s question=%s", kb.root_url, question)
            return self._fallback_answer(kb, question, ranked_pages=primary_pages)

        reply = payload.get("reply") or "Я не смог уверенно ответить по текущей карте сайта."
        suggested_url = payload.get("suggested_url") or llm_pages[0].url
        if not _same_site(suggested_url, str(kb.root_url)):
            suggested_url = llm_pages[0].url

        relevant_urls = [url for url in payload.get("relevant_pages", []) if isinstance(url, str) and _same_site(url, str(kb.root_url))]
        if not relevant_urls:
            relevant_urls = [page.url for page in llm_pages]

        return {
            "reply": reply,
            "suggested_url": suggested_url,
            "relevant_pages": relevant_urls[:3],
            "missing_inputs": payload.get("missing_inputs", []),
        }

    async def answer_question(self, kb: SiteKnowledgeBase, question: str, user_inputs: dict[str, Any]) -> dict[str, Any]:
        specific_doctor = self._specific_doctor_result(kb, question)
        if specific_doctor:
            logger.info("Specific doctor answer handled without LLM for site=%s question=%s", kb.root_url, question)
            return specific_doctor
        if self.llm.enabled:
            return await self._answer_with_llm(kb, question, user_inputs)
        logger.warning("LLM disabled in answer mode, using fallback. site=%s question=%s", kb.root_url, question)
        primary_pages, _, _ = await self._retrieve_context_bundle(
            kb,
            question,
            action_mode=False,
            primary_limit=3,
            llm_limit=3,
        )
        return self._fallback_answer(kb, question, ranked_pages=primary_pages)

    async def build_plan(self, kb: SiteKnowledgeBase, goal: str, user_inputs: dict[str, Any]) -> tuple[ActionPlan, list[PageSnapshot]]:
        ranked_pages, llm_pages, context = await self._retrieve_context_bundle(
            kb,
            goal,
            action_mode=True,
            primary_limit=8,
            llm_limit=6,
        )
        deterministic_plan, relevant_pages = self._deterministic_plan(kb, goal, user_inputs, ranked_pages=ranked_pages)
        if deterministic_plan.possible:
            logger.info("Deterministic action plan selected. site=%s goal=%s", kb.root_url, goal)
            return deterministic_plan, relevant_pages

        if not self.llm.enabled:
            logger.warning("LLM disabled in action mode, using deterministic failure. site=%s goal=%s", kb.root_url, goal)
            return deterministic_plan, relevant_pages

        fallback_url = llm_pages[0].url if llm_pages else str(kb.root_url)
        user_prompt = (
            f"Goal: {goal}\n"
            f"Known user inputs: {user_inputs}\n"
            f"Relevant site graph and page context:\n{context}"
        )
        try:
            payload = await self.llm.complete_json(PLAN_SYSTEM_PROMPT, user_prompt, request_name="build_plan")
            plan = ActionPlan.model_validate(payload)
            return self._sanitize_plan(kb, plan, fallback_url), llm_pages
        except Exception:
            logger.exception("LLM plan generation failed after deterministic miss. site=%s goal=%s", kb.root_url, goal)
            return deterministic_plan, relevant_pages

    async def handle_message(self, site_url: str, kb: SiteKnowledgeBase, message: str, user_inputs: dict[str, Any]) -> dict[str, Any]:
        collected_inputs = self._extract_inputs(message, user_inputs)
        intent = self._classify_intent(site_url, message, collected_inputs)
        recall = self._recall(site_url)

        if intent == "clarification":
            last_goal = recall.get("last_goal")
            if last_goal:
                plan, relevant_pages = await self.build_plan(kb, last_goal, collected_inputs)
                self._remember(site_url, last_goal=last_goal, last_mode="action", missing_inputs=plan.missing_inputs)
                return {
                    "mode": "clarification",
                    "collected_inputs": collected_inputs,
                    "plan": plan,
                    "reply": "Продолжаю предыдущую задачу. " + plan.explanation_for_user,
                    "suggested_url": relevant_pages[0].url if relevant_pages else str(kb.root_url),
                    "relevant_pages": [page.url for page in relevant_pages],
                }

        if intent == "answer":
            answer = await self.answer_question(kb, message, collected_inputs)
            self._remember(site_url, last_goal=None, last_mode="answer", missing_inputs=answer.get("missing_inputs", []))
            return {"mode": "answer", "collected_inputs": collected_inputs, **answer}

        plan, relevant_pages = await self.build_plan(kb, message, collected_inputs)
        self._remember(site_url, last_goal=message, last_mode="action", missing_inputs=plan.missing_inputs)
        reply = plan.explanation_for_user
        if self._is_navigation_request(message) and plan.steps and plan.steps[0].action == "goto":
            target = relevant_pages[0].title if relevant_pages else "нужный раздел"
            reply = f"Открываю {target}."
        return {
            "mode": "action",
            "collected_inputs": collected_inputs,
            "plan": plan,
            "reply": reply,
            "suggested_url": relevant_pages[0].url if relevant_pages else str(kb.root_url),
            "relevant_pages": [page.url for page in relevant_pages],
        }
