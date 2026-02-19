"""Emotion Extractor agent (Agent 2) for PCMFG.

Extracts directed emotions from text chunks:
- POV character identification
- Characters present in scene
- Directed emotion scores (9 base emotions, 1-5 scale)
- Justification quotes
"""

from typing import Any

from pydantic import ValidationError

from pcmfg.llm.base import LLMAPIError, LLMResponseParseError
from pcmfg.models.schemas import (
    BASE_EMOTIONS,
    ChunkAnalysis,
    DirectedEmotion,
    DirectedEmotionScores,
    WorldBuilderOutput,
)


def build_emotion_extractor_system_prompt(world: WorldBuilderOutput) -> str:
    """Build the system prompt for Agent 2 with world context.

    Args:
        world: World builder output containing main pairing, aliases, etc.

    Returns:
        Formatted system prompt string.
    """
    main_pairing_str = " and ".join(world.main_pairing)
    aliases_str = "\n".join(
        f"    - {name}: {', '.join(aliases)}" for name, aliases in world.aliases.items()
    )
    guidelines_str = "\n".join(f"  - {g}" for g in world.world_guidelines[:5])  # Limit to 5 key facts

    return f"""You are an expert computational literary analyst extracting granular, directed emotional data from a romance novel.

### CONTEXT
* Main Pairing: {main_pairing_str}
* Aliases:
{aliases_str}
* World Guidelines:
{guidelines_str}

### YOUR TASK
Analyze the provided text chunk and output STRICTLY in JSON.
1. Identify the "chunk_main_pov" (whose perspective we are in, or the focal character).
2. List all "characters_present" in the scene.
3. Map the DIRECTED emotions (Source -> Target) between the Main Pairing ONLY.
   - A -> B is NOT the same as B -> A.
   - Only score a direction if there is explicit textual evidence (dialogue, internal monologue, or physical action).
   - If A is thinking about B while B is absent, ONLY output the A -> B direction. Do not guess B's unwritten feelings.

### THE 9 BASE EMOTIONS (EXTENDED PLUTCHIK MODEL)
Score the Source's feelings toward the Target on each of these 9 metrics:
1. Joy (Happiness, pleasure, delight)
2. Trust (Safety, reliance, vulnerability)
3. Fear (Panic, dread, terror, anxiety)
4. Surprise (Astonishment, shock)
5. Sadness (Grief, sorrow, despair)
6. Disgust (Revulsion, aversion, contempt)
7. Anger (Fury, rage, frustration)
8. Anticipation (Looking forward to, expecting, plotting)
9. Arousal (Physical lust, romantic desire, sexual tension)

### SCORING RUBRIC (STRICT DEFAULT TO 1)
Assume 1 (Neutral/None) for ALL emotions unless explicit text proves otherwise. Normal conversation is all 1s.
* 1 (None/Baseline): No evidence of this emotion. Polite, functional, or entirely absent.
* 2 (Mild): A brief, subtle hint or low-energy flicker of the emotion.
* 3 (Moderate): Clear, undeniable presence of the emotion.
* 4 (Strong): Emotion heavily drives the character's actions or thoughts. High physiological arousal.
* 5 (Extreme): Overwhelming, consuming saturation of the emotion. Maximum intensity.

### REQUIRED JSON SCHEMA
{{
  "chunk_id": <integer>,
  "chunk_main_pov": "Name of POV character",
  "characters_present": ["Name 1", "Name 2"],
  "directed_emotions": [
    {{
      "source": "Name of Source Character",
      "target": "Name of Target Character",
      "scores": {{
        "Joy": <int 1-5>,
        "Trust": <int 1-5>,
        "Fear": <int 1-5>,
        "Surprise": <int 1-5>,
        "Sadness": <int 1-5>,
        "Disgust": <int 1-5>,
        "Anger": <int 1-5>,
        "Anticipation": <int 1-5>,
        "Arousal": <int 1-5>
      }},
      "justification_quote": "Exact text quote proving the highest active scores for this direction."
    }}
  ],
  "scene_summary": "One brief sentence summarizing the action."
}}"""


class EmotionExtractor:
    """Agent 2: Base Emotion Extractor for directed emotion scoring.

    Extracts directed emotions between main pairing characters from text chunks.
    """

    def __init__(self, llm_client: Any, world: WorldBuilderOutput) -> None:
        """Initialize Emotion Extractor.

        Args:
            llm_client: LLM client with call_json method.
            world: World builder output for context.
        """
        self.llm_client = llm_client
        self.world = world
        self._system_prompt = build_emotion_extractor_system_prompt(world)

    def extract(self, text_chunk: str, chunk_id: int, position: float = 0.0) -> ChunkAnalysis:
        """Extract directed emotions from a text chunk.

        Args:
            text_chunk: Text to analyze.
            chunk_id: Sequential chunk identifier.
            position: Position in narrative (0.0-1.0).

        Returns:
            ChunkAnalysis with extracted emotions.

        Raises:
            EmotionExtractionError: If extraction fails.
        """
        user_prompt = f"Analyze the following text chunk:\n\n{text_chunk}"

        try:
            response = self.llm_client.call_json(
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
            )

            # Parse and validate the response
            return self._parse_response(response, chunk_id, position)

        except (LLMAPIError, LLMResponseParseError) as e:
            # Return default chunk on failure
            return self._create_default_chunk(chunk_id, position, str(e))

    def _parse_response(
        self, response: dict, chunk_id: int, position: float
    ) -> ChunkAnalysis:
        """Parse and validate LLM response.

        Args:
            response: Raw JSON response from LLM.
            chunk_id: Chunk identifier.
            position: Position in narrative.

        Returns:
            Validated ChunkAnalysis.
        """
        # Ensure chunk_id is set
        response["chunk_id"] = chunk_id
        response["position"] = position

        # Parse directed emotions
        emotions_list = response.get("directed_emotions", [])
        if not isinstance(emotions_list, list):
            emotions_list = []

        parsed_emotions: list[DirectedEmotion] = []
        for emotion_data in emotions_list:
            if not isinstance(emotion_data, dict):
                continue

            parsed = self._parse_directed_emotion(emotion_data)
            if parsed is not None:
                parsed_emotions.append(parsed)

        response["directed_emotions"] = parsed_emotions

        # Ensure required fields exist
        if "chunk_main_pov" not in response:
            response["chunk_main_pov"] = self.world.main_pairing[0] if self.world.main_pairing else "Unknown"
        if "characters_present" not in response:
            response["characters_present"] = self.world.main_pairing[:2]
        if "scene_summary" not in response:
            response["scene_summary"] = ""

        try:
            return ChunkAnalysis(**response)
        except ValidationError as e:
            return self._create_default_chunk(chunk_id, position, f"Validation error: {e}")

    def _parse_directed_emotion(self, data: dict) -> DirectedEmotion | None:
        """Parse a single directed emotion from response data.

        Args:
            data: Raw emotion data dict.

        Returns:
            DirectedEmotion or None if parsing fails.
        """
        try:
            source = data.get("source", "")
            target = data.get("target", "")
            justification = data.get("justification_quote", "")

            if not source or not target:
                return None

            # Parse scores, defaulting to 1 for any missing/invalid values
            scores_data = data.get("scores", {})
            if not isinstance(scores_data, dict):
                scores_data = {}

            scores = {}
            for emotion in BASE_EMOTIONS:
                value = scores_data.get(emotion, 1)
                # Validate score is in range 1-5
                if not isinstance(value, int) or value < 1 or value > 5:
                    value = 1
                scores[emotion] = value

            return DirectedEmotion(
                source=source,
                target=target,
                scores=DirectedEmotionScores(**scores),
                justification_quote=justification,
            )

        except Exception:
            return None

    def _create_default_chunk(
        self, chunk_id: int, position: float, error_msg: str
    ) -> ChunkAnalysis:
        """Create a default chunk when extraction fails.

        Args:
            chunk_id: Chunk identifier.
            position: Position in narrative.
            error_msg: Error message for debugging.

        Returns:
            ChunkAnalysis with default/neutral values.
        """
        # Create default directed emotions (all scores = 1)
        default_emotions = []
        if len(self.world.main_pairing) >= 2:
            for source, target in [
                (self.world.main_pairing[0], self.world.main_pairing[1]),
                (self.world.main_pairing[1], self.world.main_pairing[0]),
            ]:
                default_emotions.append(
                    DirectedEmotion(
                        source=source,
                        target=target,
                        scores=DirectedEmotionScores(),  # All defaults to 1
                        justification_quote=f"[Extraction failed: {error_msg}]",
                    )
                )

        return ChunkAnalysis(
            chunk_id=chunk_id,
            position=position,
            chunk_main_pov=self.world.main_pairing[0] if self.world.main_pairing else "Unknown",
            characters_present=self.world.main_pairing[:2],
            directed_emotions=default_emotions,
            scene_summary="[Extraction failed]",
        )


class EmotionExtractionError(Exception):
    """Raised when emotion extraction fails."""

    pass
