"""llm/gemini_client.py

Wraps Gemini Flash streaming with multi-turn conversation history.
Yields text chunks as they stream from the model.

The caller owns the conversation history (a list of types.Content).
This module only adds the new user message, streams the response, and
appends the full assistant reply before returning.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Generator
from pathlib import Path

from dotenv import dotenv_values

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_THINKING_BUDGET_DISABLED = 0


@dataclass(frozen=True)
class LlmConfig:
    model: str
    system_prompt: str


class GeminiClient:
    """Stateless Gemini streaming client.

    History is stored externally in VoiceAgentOrchestrator and passed in
    on each call — this class has no mutable state.
    """

    def __init__(self, config: LlmConfig) -> None:
        # Resolve project root (parent of 'agent' folder)
        root_dir = Path(__file__).parent.parent.parent
        env_vars = dotenv_values(root_dir / ".env")
        
        api_key = env_vars.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not found in the local .env file. "
                "Ensure your .env file is in the project root."
            )
        self._client = genai.Client(api_key=api_key)
        self._config = config

    def stream_reply(
        self,
        user_text: str,
        history: list[types.Content],
    ) -> Generator[str, None, None]:
        """Stream assistant reply tokens for user_text, given history.

        Mutates `history` in-place: appends the user turn before streaming,
        then appends the full assistant reply after streaming completes.

        Yields individual text chunks (may be empty strings — skip those).
        O(n) in response length.
        """
        history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_text)],
            )
        )

        generate_config = types.GenerateContentConfig(
            system_instruction=self._config.system_prompt,
            thinking_config=types.ThinkingConfig(
                thinking_budget=_THINKING_BUDGET_DISABLED,
            ),
        )

        full_reply = ""
        for chunk in self._client.models.generate_content_stream(
            model=self._config.model,
            contents=history,
            config=generate_config,
        ):
            text = chunk.text or ""
            if text:
                full_reply += text
                yield text

        history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=full_reply)],
            )
        )
        logger.info("LLM reply (%d chars): %r", len(full_reply), full_reply[:80])
