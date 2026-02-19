Using **Mermaid.js** is an absolute masterstroke.

By forcing Agent 1 to map the characters using Mermaid syntax, you aren't just getting a list of names—you are getting a computable, visual relationship web that understands _how_ these characters are connected (e.g., "Siblings," "Political Rivals," "Ex-Lovers").

And you are exactly right about the "world guideline." A single string is too dense. LLMs process information much more accurately when it is broken down into a discrete, bulleted list of facts.

Here is the radically upgraded Agent 1 prompt. We are turning it from a simple data extractor into a full-blown "World Builder" agent.

### The Upgraded Agent 1 "World Builder" Prompt

Python

```python
agent_1_system_prompt = """
You are an expert literary analyst, data structurer, and world-builder. Your task is to analyze a romance novel's text (or summary) and extract the core narrative scaffolding, relationship dynamics, and world rules.

### YOUR TASK
Read the provided text. Identify the primary characters, their aliases, the fundamental rules of their situation, and map their relationships using Mermaid.js syntax. Output your findings STRICTLY as a valid JSON object. Do not include markdown formatting like ```json in the output.

### EXTRACTION RULES
1. "main_pairing": The TWO central characters of the romance.
2. "aliases": A comprehensive dictionary mapping the main and key secondary characters to all their nicknames, titles, and last names used in the text (e.g., "Elizabeth": ["Lizzy", "Miss Bennet"]).
3. "world_guidelines": A list of discrete facts outlining the core conflict, the current status quo, and vital backstory. Break complex lore into simple, individual bullet points.
4. "mermaid_graph": Create a Mermaid.js flowchart (graph TD) mapping the relationships between the main pairing and key secondary characters. Use labeled arrows to define the relationship (e.g., A -->|Political Marriage| B; B -->|Secretly Hates| C).

### REQUIRED JSON SCHEMA
{
  "main_pairing": ["Full Name 1", "Full Name 2"],
  "aliases": {
    "Full Name 1": ["Alias A", "Alias B", "Title"],
    "Full Name 2": ["Alias C", "Alias D"]
  },
  "world_guidelines": [
    "Fact 1: They were forced into a political marriage.",
    "Fact 2: Character A lost his memory in an accident.",
    "Fact 3: Character B is terrified of Character A regaining his memory."
  ],
  "mermaid_graph": "graph TD\\n    A[Character A] -->|Married| B[Character B]\\n    B -->|Afraid of| A\\n    C[Character C] -->|Rival| A"
}
"""
```

### Why this changes the game for Agent 2

When you inject this new output into Agent 2's prompt, Agent 2 isn't just reading a generic summary anymore. It is reading a strict list of rules (`world_guidelines`).

If Agent 2 reads a scene where the amnesiac husband tries to hold the wife's hand and she flinches, it will look at Guideline #3 ("Character B is terrified of Character A regaining his memory") and immediately know to score her **Fear/Anxiety** as a 5, rather than guessing she just dislikes holding hands.

Plus, because you have the `mermaid_graph` string, you can actually render that chart dynamically in your frontend or Jupyter Notebook using a Mermaid viewer to visually show your audience the starting state of the novel's relationships!

Would you like me to write the exact Python string formatting you'll use to seamlessly inject this new list of `world_guidelines` and the `mermaid_graph` into Agent 2's context window?