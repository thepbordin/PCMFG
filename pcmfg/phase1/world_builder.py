"""World Builder agent (Agent 1) for PCMFG.

Extracts narrative scaffolding and world context:
- Main pairing identification
- Character aliases
- World guidelines
- Relationship graph (Mermaid.js)
"""

from typing import Any

from pydantic import ValidationError

from pcmfg.llm.base import LLMAPIError, LLMResponseParseError
from pcmfg.models.schemas import WorldBuilderOutput

# System prompt for Agent 1: World Builder
WORLD_BUILDER_SYSTEM_PROMPT = """You are an expert literary analyst, data structurer, and world-builder. Your task is to analyze a romance novel's text (or summary) and extract the core narrative scaffolding, relationship dynamics, and world rules.

### YOUR TASK
Read the provided text. Identify the primary characters, their aliases, the fundamental rules of their situation, and map their relationships using Mermaid.js syntax. Output your findings STRICTLY as a valid JSON object. Do not include markdown formatting like ```json in the output.

### EXTRACTION RULES
1. "main_pairing": The TWO central characters of the romance.
2. "aliases": A comprehensive dictionary mapping the main and key secondary characters to all their nicknames, titles, and last names used in the text (e.g., "Elizabeth": ["Lizzy", "Miss Bennet"]).
3. "core_conflict": A single sentence describing the central romantic tension or obstacle between the main pairing (e.g., "Elizabeth's prejudice clashes with Darcy's pride until they overcome their misconceptions.").
4. "world_guidelines": A list of discrete facts outlining the current status quo and vital backstory. Break complex lore into simple, individual bullet points.
5. "mermaid_graph": Create a Mermaid.js flowchart (graph TD) mapping the relationships between the main pairing and key secondary characters. Use labeled arrows to define the relationship (e.g., A -->|Political Marriage| B; B -->|Secretly Hates| C).

### REQUIRED JSON SCHEMA
{
  "main_pairing": ["Full Name 1", "Full Name 2"],
  "aliases": {
    "Full Name 1": ["Alias A", "Alias B", "Title"],
    "Full Name 2": ["Alias C", "Alias D"]
  },
  "core_conflict": "A single sentence describing the central romantic tension or obstacle.",
  "world_guidelines": [
    "Fact 1: They were forced into a political marriage.",
    "Fact 2: Character A lost his memory in an accident.",
    "Fact 3: Character B is terrified of Character A regaining his memory."
  ],
  "mermaid_graph": "graph TD\\n    A[Character A] -->|Married| B[Character B]\\n    B -->|Afraid of| A"
}"""


class WorldBuilder:
    """Agent 1: World Builder for extracting narrative context.

    Analyzes text to extract:
    - Main romantic pairing
    - Character aliases
    - World guidelines (facts about the story)
    - Relationship graph (Mermaid.js)
    """

    def __init__(self, llm_client: Any) -> None:
        """Initialize World Builder.

        Args:
            llm_client: LLM client with call_json method.
        """
        self.llm_client = llm_client

    def build(self, text: str, hint: str | None = None) -> WorldBuilderOutput:
        """Build world context from text.

        Args:
            text: Input text to analyze.
            hint: Optional custom hint/instruction for non-romantic stories.
                  E.g., "This is a fairy tale. Group the three pigs as 'Pigs' and
                  identify 'Big Bad Wolf' as the antagonist. Focus on conflict, not romance."

        Returns:
            WorldBuilderOutput with extracted world information.

        Raises:
            WorldBuilderError: If extraction fails.
        """
        # Build user prompt, optionally with hint
        if hint:
            user_prompt = (
                f"SPECIAL INSTRUCTIONS:\n{hint}\n\n"
                f"Analyze the following text and extract the world information:\n\n{text}"
            )
        else:
            user_prompt = (
                f"Analyze the following text and extract the world information:\n\n{text}"
            )

        try:
            response = self.llm_client.call_json(
                system_prompt=WORLD_BUILDER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

            # Validate and parse the response
            return self._parse_response(response)

        except (LLMAPIError, LLMResponseParseError) as e:
            raise WorldBuilderError(f"Failed to extract world information: {e}")

    def _parse_response(self, response: dict) -> WorldBuilderOutput:
        """Parse and validate LLM response.

        Args:
            response: Raw JSON response from LLM.

        Returns:
            Validated WorldBuilderOutput.

        Raises:
            WorldBuilderError: If validation fails.
        """
        try:
            # Ensure main_pairing has exactly 2 characters
            main_pairing = response.get("main_pairing", [])
            if not isinstance(main_pairing, list):
                main_pairing = []
            if len(main_pairing) < 2:
                raise WorldBuilderError(
                    f"Expected 2 main characters, got {len(main_pairing)}: {main_pairing}"
                )
            # Take only first 2 if more provided
            response["main_pairing"] = main_pairing[:2]

            # Ensure aliases is a dict
            aliases = response.get("aliases", {})
            if not isinstance(aliases, dict):
                response["aliases"] = {}

            # Ensure core_conflict is a string
            core_conflict = response.get("core_conflict", "")
            if not isinstance(core_conflict, str):
                response["core_conflict"] = ""

            # Ensure world_guidelines is a list
            guidelines = response.get("world_guidelines", [])
            if not isinstance(guidelines, list):
                response["world_guidelines"] = []

            # Ensure mermaid_graph is a string
            mermaid = response.get("mermaid_graph", "")
            if not isinstance(mermaid, str):
                response["mermaid_graph"] = ""

            return WorldBuilderOutput(**response)

        except ValidationError as e:
            raise WorldBuilderError(f"Invalid world builder response: {e}")


class WorldBuilderError(Exception):
    """Raised when world builder extraction fails."""

    pass
