from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from core.config import get_settings


logger = logging.getLogger("web_agent.llm")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.openrouter_api_key)

    async def complete_json(self, system_prompt: str, user_prompt: str, *, request_name: str = "unknown") -> dict[str, Any]:
        if not self.enabled:
            logger.warning("LLM disabled for %s: OPENROUTER_API_KEY is not configured.", request_name)
            raise RuntimeError("OPENROUTER_API_KEY is not configured")

        url = f"{self.settings.openrouter_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.settings.frontend_origin,
            "X-Title": "Universal Website Agent",
        }

        logger.info(
            "OPENROUTER REQUEST [%s]\nmodel=%s\nurl=%s\n--- SYSTEM PROMPT START ---\n%s\n--- SYSTEM PROMPT END ---\n--- USER PROMPT START ---\n%s\n--- USER PROMPT END ---",
            request_name,
            self.settings.llm_model,
            url,
            system_prompt,
            user_prompt,
        )

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(url, headers=headers, json=payload)
                logger.info(
                    "OPENROUTER RESPONSE [%s] status=%s body=%s",
                    request_name,
                    response.status_code,
                    response.text[:2000],
                )
                response.raise_for_status()
        except httpx.HTTPError:
            logger.exception("OPENROUTER HTTP ERROR [%s]", request_name)
            raise
        except Exception:
            logger.exception("OPENROUTER UNEXPECTED ERROR [%s]", request_name)
            raise

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            text = content
        logger.info("OPENROUTER PARSED CONTENT [%s] %s", request_name, text[:2000])
        return json.loads(text)
