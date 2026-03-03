from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agent.service import AgentService
from core.config import get_settings
from core.models import (
    ChatRequest,
    ChatResponse,
    CreateAgentResponse,
    CrawlRequest,
    CrawlResponse,
    ExecuteRequest,
    PlanRequest,
)
from crawler.service import SiteCrawler
from executor.service import BrowserExecutor
from kb.storage import KnowledgeBaseStore


app = FastAPI(title="Universal Website Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = KnowledgeBaseStore()
crawler = SiteCrawler()
agent = AgentService()
executor = BrowserExecutor()
settings = get_settings()

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
app.mount("/artifacts", StaticFiles(directory=store.base_dir), name="artifacts")


def _preview_for_url(kb, url: str | None) -> str | None:
    if not url:
        return None
    for page in kb.pages:
        if page.url == url and page.preview_image_url:
            return page.preview_image_url
    for page in kb.pages:
        if page.url == str(kb.root_url) and page.preview_image_url:
            return page.preview_image_url
    return kb.pages[0].preview_image_url if kb.pages else None


def _allows_frame_ancestors(csp: str, frontend_origin: str) -> bool:
    directives = [directive.strip() for directive in csp.split(";") if directive.strip()]
    frame_directive = next((directive for directive in directives if directive.lower().startswith("frame-ancestors")), "")
    if not frame_directive:
        return True

    allowed = frame_directive.split()[1:]
    if not allowed:
        return False
    if "*" in allowed:
        return True

    frontend_host = urlparse(frontend_origin).netloc
    frontend_origin_lower = frontend_origin.lower()
    for item in allowed:
        normalized = item.strip().strip("'").lower()
        if normalized in {"self", "none"}:
            continue
        if frontend_origin_lower in normalized or frontend_host in normalized:
            return True
    return False


async def _live_preview_allowed(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
            try:
                response = await client.head(url)
            except Exception:
                response = await client.get(url)
    except Exception:
        return False

    x_frame_options = (response.headers.get("x-frame-options") or "").lower()
    if x_frame_options:
        if "deny" in x_frame_options:
            return False
        if "sameorigin" in x_frame_options:
            return False

    csp = response.headers.get("content-security-policy") or ""
    return _allows_frame_ancestors(csp, settings.frontend_origin)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html", headers={"Cache-Control": "no-store"})


@app.post("/api/create-agent", response_model=CreateAgentResponse)
async def create_agent(payload: CrawlRequest) -> CreateAgentResponse:
    kb = await crawler.crawl(str(payload.url), max_pages=payload.max_pages)
    kb = await agent.enrich_kb_embeddings(kb)
    path = store.save_kb(kb)
    domain = urlparse(str(payload.url)).netloc
    preview_url = kb.pages[0].url if kb.pages else str(payload.url)
    live_preview_allowed = await _live_preview_allowed(preview_url)
    return CreateAgentResponse(
        agent_name=f"Agent for {domain}",
        site_url=preview_url,
        kb_path=str(path),
        pages_discovered=len(kb.pages),
        transition_count=len(kb.transitions),
        greeting=(
            f"Агент для {domain} готов. В этом чате можно либо попросить выполнить действие на сайте, "
            "либо просто задать вопрос по структуре и разделам сайта."
        ),
        preview_image_url=_preview_for_url(kb, preview_url),
        live_preview_allowed=live_preview_allowed,
    )


@app.post("/api/crawl", response_model=CrawlResponse)
async def crawl_site(payload: CrawlRequest) -> CrawlResponse:
    kb = await crawler.crawl(str(payload.url), max_pages=payload.max_pages)
    kb = await agent.enrich_kb_embeddings(kb)
    path = store.save_kb(kb)
    return CrawlResponse(
        kb_path=str(path),
        pages_discovered=len(kb.pages),
        transition_count=len(kb.transitions),
    )


@app.post("/api/plan")
async def create_plan(payload: PlanRequest):
    kb = store.load_kb(str(payload.url))
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found. Crawl the site first.")
    plan, relevant_pages = await agent.build_plan(kb, payload.user_goal, payload.user_inputs)
    plan_path = store.save_plan(str(payload.url), plan)
    return {
        "plan": plan.model_dump(mode="json"),
        "plan_path": str(plan_path),
        "relevant_pages": [page.url for page in relevant_pages],
    }


@app.post("/api/execute")
async def execute(payload: ExecuteRequest):
    preview_path, preview_web_path = store.runtime_preview_path(str(payload.url))
    result = await executor.execute_plan(
        str(payload.url),
        payload.plan,
        payload.user_inputs,
        preview_path=preview_path,
        preview_web_path=preview_web_path,
    )
    log_path = store.save_execution_log(str(payload.url), result)
    result.log_path = str(log_path)
    return result.model_dump(mode="json")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    kb = store.load_kb(str(payload.url))
    if not kb:
        raise HTTPException(status_code=404, detail="Agent not found. Create the agent first.")

    response = await agent.handle_message(str(payload.url), kb, payload.message, payload.user_inputs)
    plan = response.get("plan")
    plan_path = store.save_plan(str(payload.url), plan) if plan else None

    if response["mode"] == "answer":
        suggested_url = response.get("suggested_url")
        return ChatResponse(
            mode="answer",
            reply=response["reply"],
            collected_inputs=response.get("collected_inputs", {}),
            suggested_url=suggested_url,
            preview_image_url=_preview_for_url(kb, suggested_url),
            live_preview_allowed=await _live_preview_allowed(suggested_url) if suggested_url else False,
            relevant_pages=response.get("relevant_pages", []),
            plan=None,
            execution=None,
            plan_path=None,
            log_path=None,
        )

    if not plan:
        raise HTTPException(status_code=500, detail="Agent did not return a plan for action mode.")

    if not plan.possible or plan.missing_inputs:
        suggested_url = response.get("suggested_url")
        return ChatResponse(
            mode=response["mode"],
            reply=response["reply"],
            collected_inputs=response.get("collected_inputs", {}),
            suggested_url=suggested_url,
            preview_image_url=_preview_for_url(kb, suggested_url),
            live_preview_allowed=await _live_preview_allowed(suggested_url) if suggested_url else False,
            relevant_pages=response.get("relevant_pages", []),
            plan=plan,
            execution=None,
            plan_path=str(plan_path) if plan_path else None,
            log_path=None,
        )

    preview_path, preview_web_path = store.runtime_preview_path(str(payload.url))
    result = await executor.execute_plan(
        str(payload.url),
        plan,
        response.get("collected_inputs", {}),
        preview_path=preview_path,
        preview_web_path=preview_web_path,
    )
    log_path = store.save_execution_log(str(payload.url), result)
    result.log_path = str(log_path)

    reply = response["reply"]
    if result.stopped_for_human:
        reply += " Проверьте, пожалуйста, поля на сайте и завершите финальное действие самостоятельно."
    elif result.success:
        reply += " Действие выполнено до безопасной точки."
    elif not result.success:
        reply += " Во время исполнения возникла ошибка. Посмотрите лог шага."

    final_url = result.current_url or response.get("suggested_url")
    return ChatResponse(
        mode=response["mode"],
        reply=reply,
        collected_inputs=response.get("collected_inputs", {}),
        suggested_url=final_url,
        preview_image_url=result.preview_image_url or _preview_for_url(kb, final_url),
        live_preview_allowed=await _live_preview_allowed(final_url) if final_url else False,
        relevant_pages=response.get("relevant_pages", []),
        plan=plan,
        execution=result,
        plan_path=str(plan_path) if plan_path else None,
        log_path=str(log_path),
    )


@app.get("/api/kb")
async def get_kb(url: str):
    kb = store.load_kb(url)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found.")
    return kb.model_dump(mode="json")
