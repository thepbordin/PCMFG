"""Base LLM client interface for PCMFG.

This module defines the protocol that all LLM clients must implement,
providing a consistent interface for text and JSON generation.
"""

import json
from abc import abstractmethod
from typing import Protocol


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class LLMRateLimitError(LLMError):
    """Raised when LLM API rate limit is hit."""

    pass


class LLMAPIError(LLMError):
    """Raised when LLM API call fails."""

    pass


class LLMResponseParseError(LLMError):
    """Raised when LLM response cannot be parsed."""

    pass


class BaseLLMClient(Protocol):
    """Protocol for LLM client implementations.

    All LLM clients (OpenAI, Anthropic, etc.) must implement this protocol
    to be used with the PCMFG pipeline.
    """

    @abstractmethod
    def call(self, system_prompt: str, user_prompt: str) -> str:
        """Make a text generation call to the LLM.

        Args:
            system_prompt: System prompt for the LLM.
            user_prompt: User prompt for the LLM.

        Returns:
            Generated text response.

        Raises:
            LLMRateLimitError: If rate limit is hit.
            LLMAPIError: If API call fails.
        """
        ...

    @abstractmethod
    def call_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Make a JSON generation call to the LLM.

        The LLM is instructed to return valid JSON, which is parsed and returned.

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
        ...


def parse_json_response(response: str) -> dict:
    """Parse JSON from an LLM response.

    Handles common issues:
    - Markdown code blocks (```json ... ```)
    - Trailing/leading whitespace
    - Missing closing braces

    Args:
        response: Raw LLM response text.

    Returns:
        Parsed JSON as dictionary.

    Raises:
        LLMResponseParseError: If JSON cannot be parsed.
    """
    text = response.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        # Find the end of the code block
        lines = text.split("\n")
        # Remove first line (```json or ```)
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line (```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return dict(json.loads(text))
    except json.JSONDecodeError as e:
        raise LLMResponseParseError(f"Failed to parse JSON response: {e}\nResponse: {text[:500]}")
