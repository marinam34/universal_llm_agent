from __future__ import annotations

import json
from collections import Counter
from math import ceil
from pathlib import Path
from urllib.parse import urlparse

from core.config import get_settings
from core.models import ActionPlan, ExecutionResult, SiteKnowledgeBase


class KnowledgeBaseStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_dir = self.settings.artifacts_dir

    def _site_dir(self, site_url: str) -> Path:
        parsed = urlparse(site_url)
        netloc = parsed.netloc.replace(":", "_") or "site"
        site_dir = self.base_dir / netloc
        site_dir.mkdir(parents=True, exist_ok=True)
        return site_dir

    def save_kb(self, kb: SiteKnowledgeBase) -> Path:
        site_dir = self._site_dir(str(kb.root_url))
        path = site_dir / "knowledge_base.json"
        path.write_text(json.dumps(kb.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        graph_path = site_dir / "site_graph.json"
        compact_edges, global_edges = self._compact_edges(kb)
        focused_edges = self._focused_edges(kb, compact_edges)
        outgoing = Counter(edge["from_url"] for edge in focused_edges)
        incoming = Counter(edge["to_url"] for edge in focused_edges)
        graph_payload = {
            "root_url": str(kb.root_url),
            "domain": kb.domain,
            "meta": {
                "page_count": len(kb.pages),
                "focused_edge_count": len(focused_edges),
                "compact_edge_count": len(compact_edges),
                "global_edge_count": len(global_edges),
            },
            "nodes": [
                {
                    "url": page.url,
                    "title": page.title,
                    "summary": page.summary,
                    "tags": page.tags,
                    "outgoing_count": outgoing.get(page.url, 0),
                    "incoming_count": incoming.get(page.url, 0),
                }
                for page in kb.pages
            ],
            "edges": focused_edges,
            "global_edges": global_edges,
        }
        graph_path.write_text(json.dumps(graph_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _compact_edges(self, kb: SiteKnowledgeBase) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        aggregated: dict[tuple[str, str], dict[str, object]] = {}
        valid_urls = {page.url for page in kb.pages}
        page_by_url = {page.url: page for page in kb.pages}
        root_url = str(kb.root_url)

        for edge in kb.transitions:
            if edge.from_url == edge.to_url:
                continue
            if edge.from_url not in valid_urls or edge.to_url not in valid_urls:
                continue

            key = (edge.from_url, edge.to_url)
            item = aggregated.get(key)
            if not item:
                aggregated[key] = {
                    "from_url": edge.from_url,
                    "to_url": edge.to_url,
                    "trigger_type": edge.trigger_type,
                    "trigger_text": edge.trigger_text,
                    "selector": edge.selector,
                    "occurrence_count": 1,
                }
                continue

            item["occurrence_count"] = int(item["occurrence_count"]) + 1
            current_text = (item.get("trigger_text") or "").strip()
            new_text = (edge.trigger_text or "").strip()
            if new_text and (not current_text or len(new_text) < len(current_text)):
                item["trigger_text"] = new_text
            current_selector = (item.get("selector") or "").strip()
            if not current_selector and edge.selector:
                item["selector"] = edge.selector

        unique_edges = list(aggregated.values())
        incoming_frequency = Counter(edge["to_url"] for edge in unique_edges)
        page_count = max(len(kb.pages), 1)
        global_threshold = max(8, ceil(page_count * 0.3))
        important_tags = {"has_form", "lead_form", "booking", "doctors"}

        compact_edges: list[dict[str, object]] = []
        global_edges: list[dict[str, object]] = []
        for edge in unique_edges:
            target_page = page_by_url.get(str(edge["to_url"]))
            target_tags = set(target_page.tags) if target_page else set()
            is_global_nav = (
                edge["trigger_type"] == "link"
                and incoming_frequency[str(edge["to_url"])] >= global_threshold
                and not (target_tags & important_tags)
                and str(edge["from_url"]) != root_url
            )
            if is_global_nav:
                global_edges.append(edge)
            else:
                compact_edges.append(edge)

        compact_edges.sort(key=lambda item: (item["from_url"], item["to_url"]))
        global_edges.sort(key=lambda item: (item["from_url"], item["to_url"]))
        return compact_edges, global_edges

    def _focused_edges(self, kb: SiteKnowledgeBase, compact_edges: list[dict[str, object]]) -> list[dict[str, object]]:
        page_by_url = {page.url: page for page in kb.pages}
        grouped: dict[str, list[dict[str, object]]] = {}
        important_tags = {"has_form", "lead_form", "booking", "doctors", "services", "contacts"}
        root_url = str(kb.root_url)

        for edge in compact_edges:
            grouped.setdefault(str(edge["from_url"]), []).append(edge)

        focused: list[dict[str, object]] = []
        for from_url, edges in grouped.items():
            def edge_score(edge: dict[str, object]) -> tuple[int, int, int]:
                target_page = page_by_url.get(str(edge["to_url"]))
                target_tags = set(target_page.tags) if target_page else set()
                path = urlparse(str(edge["to_url"])).path.strip("/")
                depth = len([part for part in path.split("/") if part])
                score = 0
                if edge.get("trigger_type") == "form":
                    score += 8
                elif edge.get("trigger_type") == "button":
                    score += 5
                score += min(int(edge.get("occurrence_count", 1)), 4)
                score += len(target_tags & important_tags) * 3
                score += max(0, 4 - depth)
                if from_url == root_url and depth > 2:
                    score -= (depth - 2) * 3
                trigger_text = str(edge.get("trigger_text") or "").strip()
                if 3 <= len(trigger_text) <= 80:
                    score += 1
                return (score, len(target_tags), int(edge.get("occurrence_count", 1)))

            limit = 18 if from_url == root_url else 8
            ranked = sorted(edges, key=edge_score, reverse=True)
            focused.extend(ranked[:limit])

        focused.sort(key=lambda item: (item["from_url"], item["to_url"]))
        return focused

    def load_kb(self, site_url: str) -> SiteKnowledgeBase | None:
        path = self._site_dir(site_url) / "knowledge_base.json"
        if not path.exists():
            return None
        return SiteKnowledgeBase.model_validate_json(path.read_text(encoding="utf-8"))

    def save_plan(self, site_url: str, plan: ActionPlan) -> Path:
        site_dir = self._site_dir(site_url)
        path = site_dir / "action_plan.json"
        path.write_text(json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def save_execution_log(self, site_url: str, result: ExecutionResult) -> Path:
        site_dir = self._site_dir(site_url)
        path = site_dir / "execution_log.json"
        path.write_text(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def runtime_preview_path(self, site_url: str) -> tuple[Path, str]:
        site_dir = self._site_dir(site_url)
        path = site_dir / "runtime_preview.png"
        web_path = f"/artifacts/{site_dir.name}/runtime_preview.png"
        return path, web_path
