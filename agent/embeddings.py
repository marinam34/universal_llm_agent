from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from core.config import get_settings


logger = logging.getLogger("web_agent.embeddings")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class EmbeddingClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def enabled(self) -> bool:
        return bool(self.settings.openrouter_api_key and self.settings.embedding_model)

    async def embed_texts(self, texts: list[str], *, request_name: str = "embeddings") -> list[list[float]]:
        if not self.enabled:
            logger.warning("Embeddings disabled for %s: API key or embedding model is missing.", request_name)
            raise RuntimeError("Embeddings are not configured")

        url = f"{self.settings.openrouter_base_url.rstrip('/')}/embeddings"
        payload = {
            "model": self.settings.embedding_model,
            "input": texts,
            "encoding_format": "float",
        }
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.settings.frontend_origin,
            "X-Title": "Universal Website Agent",
        }

        logger.info(
            "OPENROUTER EMBEDDINGS REQUEST [%s]\nmodel=%s\nurl=%s\ninput_count=%s\nsample=%s",
            request_name,
            self.settings.embedding_model,
            url,
            len(texts),
            json.dumps(texts[:2], ensure_ascii=False)[:2000],
        )
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                response = await client.post(url, headers=headers, json=payload)
                logger.info(
                    "OPENROUTER EMBEDDINGS RESPONSE [%s] status=%s body=%s",
                    request_name,
                    response.status_code,
                    response.text[:2000],
                )
                response.raise_for_status()
        except httpx.HTTPError:
            logger.exception("OPENROUTER EMBEDDINGS HTTP ERROR [%s]", request_name)
            raise
        except Exception:
            logger.exception("OPENROUTER EMBEDDINGS UNEXPECTED ERROR [%s]", request_name)
            raise

        data = response.json().get("data", [])
        vectors: list[list[float]] = []
        for item in data:
            embedding = item.get("embedding")
            if isinstance(embedding, list):
                vectors.append([float(value) for value in embedding])
        return vectors
