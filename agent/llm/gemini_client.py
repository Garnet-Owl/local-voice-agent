from dataclasses import dataclass
from typing import AsyncGenerator

from google import genai
from google.genai import types

from shared.logging import setup_logging

logger = setup_logging(__name__)


@dataclass(frozen=True)
class LlmConfig:
    model: str
    system_prompt: str
    api_key: str | None = None


class GeminiClient:
    def __init__(self, config: LlmConfig) -> None:
        if not config.api_key:
            raise EnvironmentError("GEMINI_API_KEY not provided in configuration.")

        self._client = genai.Client(api_key=config.api_key)
        self._config = config
        self._generate_config = types.GenerateContentConfig(
            system_instruction=self._config.system_prompt,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        logger.info(f"LLM Client initialized with model: {config.model}")

    def new_chat(self) -> genai.chats.AsyncChat:
        return self._client.aio.chats.create(
            model=self._config.model,
            config=self._generate_config,
        )

    async def stream_reply(
        self,
        user_text: str,
        chat: genai.chats.AsyncChat,
    ) -> AsyncGenerator[str, None]:
        async for chunk in await chat.send_message_stream(user_text):
            text = chunk.text or ""
            if text:
                yield text
