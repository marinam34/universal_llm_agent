from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from core.config import get_settings
from core.models import ActionPlan, ActionStep, ExecutionResult, ExecutionStepLog


BLOCKED_TERMS = ("submit", "confirm", "подтверд", "оплат", "pay", "delete", "удал")


@dataclass
class BrowserSession:
    playwright: Playwright
    browser: Browser
    context: BrowserContext
    page: Page


class BrowserExecutor:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._sessions: dict[str, BrowserSession] = {}

    async def _start_session(self, site_key: str) -> BrowserSession:
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=self.settings.browser_headless)
        context = await browser.new_context(viewport={"width": 1440, "height": 960})
        page = await context.new_page()
        session = BrowserSession(playwright=pw, browser=browser, context=context, page=page)
        self._sessions[site_key] = session
        return session

    async def _get_session(self, site_key: str) -> BrowserSession:
        session = self._sessions.get(site_key)
        if session:
            return session
        return await self._start_session(site_key)

    async def close_session(self, site_key: str) -> None:
        session = self._sessions.pop(site_key, None)
        if not session:
            return
        await session.context.close()
        await session.browser.close()
        await session.playwright.stop()

    async def close(self) -> None:
        for site_key in list(self._sessions):
            await self.close_session(site_key)

    def _resolve_value(self, value: str | None, user_inputs: dict[str, Any]) -> str | None:
        if not value:
            return value
        resolved = value
        for key, raw in user_inputs.items():
            resolved = resolved.replace(f"{{{{{key}}}}}", str(raw))
        return resolved

    async def _capture_preview(self, page: Page, preview_path: Path | None) -> None:
        if not preview_path:
            return
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(preview_path), full_page=True)

    async def _verify(self, page: Page, step: ActionStep) -> str:
        if step.expected_url_contains and step.expected_url_contains not in page.url:
            raise RuntimeError(f"URL did not change as expected: {page.url}")
        if step.expected_selector:
            await page.wait_for_selector(step.expected_selector, timeout=5000)
        if step.expected_text:
            await page.wait_for_selector(f"text={step.expected_text}", timeout=5000)
        error_locator = page.locator("text=/error|ошибка|invalid/i")
        if await error_locator.count():
            raise RuntimeError("Detected possible error message on the page")
        return page.url

    def _is_blocked(self, step: ActionStep) -> bool:
        text = " ".join([step.selector or "", step.reason or "", step.text or ""]).lower()
        return step.action == "click" and any(term in text for term in BLOCKED_TERMS)

    async def _run_step(self, page: Page, step: ActionStep, user_inputs: dict[str, Any]) -> str:
        if step.action == "goto":
            if not step.url:
                raise RuntimeError("goto step requires url")
            await page.goto(step.url, wait_until="domcontentloaded")
        elif step.action == "click":
            if not step.selector:
                raise RuntimeError("click step requires selector")
            await page.wait_for_selector(step.selector, timeout=5000)
            await page.click(step.selector)
        elif step.action == "type":
            if not step.selector:
                raise RuntimeError("type step requires selector")
            await page.wait_for_selector(step.selector, timeout=5000)
            await page.fill(step.selector, self._resolve_value(step.value, user_inputs) or "")
        elif step.action == "select":
            if not step.selector:
                raise RuntimeError("select step requires selector")
            await page.wait_for_selector(step.selector, timeout=5000)
            option = self._resolve_value(step.option or step.value, user_inputs)
            try:
                await page.select_option(step.selector, label=option)
            except Exception:
                await page.select_option(step.selector, value=option)
        elif step.action == "wait_for":
            if step.selector:
                await page.wait_for_selector(step.selector, timeout=8000)
            elif step.text:
                await page.wait_for_selector(f"text={step.text}", timeout=8000)
            else:
                await page.wait_for_timeout(1000)
        else:
            raise RuntimeError(f"Unsupported action: {step.action}")

        await page.wait_for_timeout(500)
        return await self._verify(page, step)

    def _result_with_preview(
        self,
        *,
        success: bool,
        stopped_for_human: bool,
        current_url: str | None,
        logs: list[ExecutionStepLog],
        preview_web_path: str | None,
    ) -> ExecutionResult:
        cache_busted = f"{preview_web_path}?v={int(time.time() * 1000)}" if preview_web_path else None
        return ExecutionResult(
            success=success,
            stopped_for_human=stopped_for_human,
            current_url=current_url,
            preview_image_url=cache_busted,
            logs=logs,
        )

    async def execute_plan(
        self,
        site_url: str,
        plan: ActionPlan,
        user_inputs: dict[str, Any],
        *,
        preview_path: Path | None = None,
        preview_web_path: str | None = None,
    ) -> ExecutionResult:
        session = await self._get_session(site_url)
        page = session.page
        logs: list[ExecutionStepLog] = []

        try:
            for step in plan.steps:
                if self._is_blocked(step):
                    logs.append(
                        ExecutionStepLog(
                            step=step,
                            status="blocked",
                            detail="Blocked irreversible action.",
                            current_url=page.url,
                        )
                    )
                    await self._capture_preview(page, preview_path)
                    return self._result_with_preview(
                        success=False,
                        stopped_for_human=True,
                        current_url=page.url,
                        logs=logs,
                        preview_web_path=preview_web_path,
                    )

                try:
                    current_url = await self._run_step(page, step, user_inputs)
                    logs.append(ExecutionStepLog(step=step, status="success", detail="Step executed.", current_url=current_url))
                except Exception as exc:
                    logs.append(ExecutionStepLog(step=step, status="failed", detail=str(exc), current_url=page.url))
                    await self._capture_preview(page, preview_path)
                    return self._result_with_preview(
                        success=False,
                        stopped_for_human=False,
                        current_url=page.url,
                        logs=logs,
                        preview_web_path=preview_web_path,
                    )

            if plan.stop_before:
                stop_step = plan.stop_before[0]
                logs.append(
                    ExecutionStepLog(
                        step=stop_step,
                        status="blocked",
                        detail="Stopped before irreversible step. User must complete it manually.",
                        current_url=page.url,
                    )
                )
                await self._capture_preview(page, preview_path)
                return self._result_with_preview(
                    success=True,
                    stopped_for_human=True,
                    current_url=page.url,
                    logs=logs,
                    preview_web_path=preview_web_path,
                )

            await self._capture_preview(page, preview_path)
            return self._result_with_preview(
                success=True,
                stopped_for_human=False,
                current_url=page.url,
                logs=logs,
                preview_web_path=preview_web_path,
            )
        except Exception:
            await self._capture_preview(page, preview_path)
            raise
