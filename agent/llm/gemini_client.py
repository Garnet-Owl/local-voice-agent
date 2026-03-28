import os
from dataclasses import dataclass
from typing import AsyncGenerator

from google import genai
from google.genai import types

from shared.logging import setup_logging

logger = setup_logging("gemini_client")


@dataclass(frozen=True)
class LlmConfig:
    model: str
    system_prompt: str


class GeminiClient:
    def __init__(self, config: LlmConfig) -> None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable not set.")

        self._client = genai.Client(api_key=api_key)
        self._config = config
        logger.info(f"LLM Client initialized with model: {config.model}")

    async def stream_reply(
        self,
        user_text: str,
        history: list[types.Content],
    ) -> AsyncGenerator[str, None]:
        history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_text)])
        )

        generate_config = types.GenerateContentConfig(
            system_instruction=self._config.system_prompt,
        )

        full_reply = ""
        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self._config.model,
            contents=history,
            config=generate_config,
        ):
            text = chunk.text or ""
            if text:
                full_reply += text
                yield text

        history.append(
            types.Content(role="model", parts=[types.Part.from_text(text=full_reply)])
        )
