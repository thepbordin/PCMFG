"""OpenAI client implementation for PCMFG."""

import os
from typing import Any

from openai import APIError, RateLimitError
from openai import OpenAI as OpenAIClientSDK
from tenacity import retry, stop_after_attempt, wait_exponential

from pcmfg.llm.base import (
    LLMAPIError,
    LLMRateLimitError,
    LLMResponseParseError,
    parse_json_response,
)


class OpenAIClient:
    """OpenAI API client for PCMFG.

    Wraps the OpenAI API with retry logic and JSON response handling.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model name to use.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens per response.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = OpenAIClientSDK(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def call(self, system_prompt: str, user_prompt: str) -> str:
        """Make a text generation call to OpenAI.

        Args:
            system_prompt: System prompt for the LLM.
            user_prompt: User prompt for the LLM.

        Returns:
            Generated text response.

        Raises:
            LLMRateLimitError: If rate limit is hit.
            LLMAPIError: If API call fails.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content
            if content is None:
                raise LLMAPIError("OpenAI returned empty response")

            return content

        except RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}")
        except APIError as e:
            raise LLMAPIError(f"OpenAI API error: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def call_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Make a JSON generation call to OpenAI.

        Uses response_format={"type": "json_object"} for reliable JSON output.

        Args:
            system_prompt: System prompt for the LLM.
            user_prompt: User prompt for the LLM.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            LLMRateLimitError: If rate limit is hit.
            LLMAPIError: If API call fails.
            LLMResponseParseError: If response cannot be parsed as JSON.
        """
        try:
            # Ensure system prompt mentions JSON
            json_system_prompt = system_prompt
            if "JSON" not in json_system_prompt.upper():
                json_system_prompt += "\n\nYou must respond with valid JSON only."

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": json_system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if content is None:
                raise LLMAPIError("OpenAI returned empty response")

            return parse_json_response(content)

        except RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit exceeded: {e}")
        except APIError as e:
            raise LLMAPIError(f"OpenAI API error: {e}")
        except LLMResponseParseError:
            raise
        except Exception as e:
            raise LLMAPIError(f"Unexpected error: {e}")
