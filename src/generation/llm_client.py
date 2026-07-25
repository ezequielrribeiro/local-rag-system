import logging

import requests

from src.generation.prompts import PROMPT_TEMPLATE, get_system_prompt
from src.models import DocumentChunk

logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = "Informação não encontrada na documentação técnica fornecida."


class LLMClient:
    def __init__(
        self,
        endpoint: str = "http://localhost:11434/api/chat",
        model: str = "llama3.2",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self.endpoint = endpoint
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        query: str,
        context_chunks: list[DocumentChunk],
        doc_type: str = "tech",
    ) -> str:
        if not context_chunks:
            return NO_CONTEXT_MESSAGE

        context_parts = []
        for i, chunk in enumerate(context_chunks, start=1):
            source = chunk.metadata.source
            filename = chunk.metadata.filename
            context_parts.append(
                f"[{i}] Source: {source} | File: {filename}\n{chunk.page_content}"
            )
        retrieved_context = "\n\n---\n\n".join(context_parts)

        user_prompt = PROMPT_TEMPLATE.format(
            retrieved_context=retrieved_context,
            user_query=query,
        )
        system_prompt = get_system_prompt(doc_type)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "stream": False,
        }

        try:
            resp = requests.post(self.endpoint, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("message", {}).get("content", "")
            return answer.strip()
        except requests.exceptions.ConnectionError:
            logger.error(
                "Cannot connect to Ollama at %s. Is it running?", self.endpoint
            )
            return NO_CONTEXT_MESSAGE
        except Exception:
            logger.exception("Ollama request failed")
            return NO_CONTEXT_MESSAGE
