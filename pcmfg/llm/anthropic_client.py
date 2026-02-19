"""Anthropic Claude client implementation for PCMFG."""

import os
from typing import Any

from anthropic import Anthropic as AnthropicClient
from anthropic import APIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

from pcmfg.llm.base import (
    LLMAPIError,
    LLMRateLimitError,
    LLMResponseParseError,
    parse_json_response,
)


class AnthropicLLMClient:
    """Anthropic Claude API client for PCMFG.

    Wraps the Anthropic API with retry logic and JSON response handling.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize Anthropic client.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
            model: Model name to use.
            temperature: Sampling temperature (0.0-1.0 for Anthropic).
            max_tokens: Maximum tokens per response.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = AnthropicClient(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def call(self, system_prompt: str, user_prompt: str) -> str:
        """Make a text generation call to Anthropic.

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
            response = self._client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract text from response blocks
            text_content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text_content += block.text

            if not text_content:
                raise LLMAPIError("Anthropic returned empty response")

            return text_content

        except RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic rate limit exceeded: {e}")
        except APIError as e:
            raise LLMAPIError(f"Anthropic API error: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def call_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Make a JSON generation call to Anthropic.

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
        # Ensure system prompt mentions JSON
        json_system_prompt = system_prompt
        if "JSON" not in json_system_prompt.upper():
            json_system_prompt += (
                "\n\nYou must respond with valid JSON only. "
                "Do not include any text outside the JSON structure."
            )

        try:
            response = self._client.messages.create(
                model=self.model,
                system=json_system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract text from response blocks
            text_content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text_content += block.text

            if not text_content:
                raise LLMAPIError("Anthropic returned empty response")

            return parse_json_response(text_content)

        except RateLimitError as e:
            raise LLMRateLimitError(f"Anthropic rate limit exceeded: {e}")
        except APIError as e:
            raise LLMAPIError(f"Anthropic API error: {e}")
        except LLMResponseParseError:
            raise
        except Exception as e:
            raise LLMAPIError(f"Unexpected error: {e}")
