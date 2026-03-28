from dataclasses import dataclass
from typing import AsyncGenerator
from pathlib import Path
from dotenv import dotenv_values
from google import genai
from google.genai import types


@dataclass(frozen=True)
class LlmConfig:
    model: str
    system_prompt: str


class GeminiClient:
    """Async, stateless, fast Gemini streaming client."""

    def __init__(self, config: LlmConfig) -> None:
        root_dir = Path(__file__).parent.parent.parent
        env_vars = dotenv_values(root_dir / ".env")
        api_key = env_vars.get("GEMINI_API_KEY")

        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not found in the local .env file.")

        self._client = genai.Client(api_key=api_key)
        self._config = config

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
        # Use the async client (.aio) to stream without blocking
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
