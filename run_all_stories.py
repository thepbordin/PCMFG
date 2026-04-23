"""Run PCMFG analysis on all stories with dynamic chunk sizing.

Strategy:
- Target ~100 data points per story for good Matrix Profile quality
- Compute max_chunk_tokens dynamically from total word count
- Save raw results as JSON, then run interesting section detection
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
BASE_CONFIG = Path("pcmfg_config.yaml")

TARGET_CHUNKS = 50
WORDS_PER_TOKEN = 1.3
MIN_CHUNK_TOKENS = 500
MIN_CHUNKS_THRESHOLD = 30  # skip re-run if already has this many chunks


def get_text_for_story(story_dir: Path) -> str:
    """Merge all text files from a story directory."""
    subdirs = sorted([d for d in story_dir.iterdir() if d.is_dir()])
    if subdirs:
        parts = []
        for subdir in subdirs:
            for f in sorted(subdir.iterdir()):
                if f.is_file() and f.suffix in (".md", ".txt"):
                    parts.append(f.read_text(encoding="utf-8"))
        return "\n\n".join(parts)
    else:
        files = sorted(
            list(story_dir.glob("*.md")) + list(story_dir.glob("*.txt"))
        )
        return "\n\n".join(f.read_text(encoding="utf-8") for f in files)


def compute_chunk_tokens(total_words: int) -> int:
    """Compute max_chunk_tokens to hit ~TARGET_CHUNKS data points."""
    if total_words == 0:
        return MIN_CHUNK_TOKENS
    target_words_per_chunk = total_words / TARGET_CHUNKS
    return max(MIN_CHUNK_TOKENS, int(target_words_per_chunk * WORDS_PER_TOKEN))


def make_config(chunk_tokens: int) -> Path:
    """Write a temporary config with the given max_chunk_tokens, keeping
    everything else from the base config."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["processing"]["max_chunk_tokens"] = chunk_tokens
    cfg["processing"]["beat_detection"] = "length"
    cfg_path = Path("_dynamic_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return cfg_path


def main():
    # Load base config to show model info
    with open(BASE_CONFIG) as f:
        base_cfg = yaml.safe_load(f)
    model = base_cfg.get("llm", {}).get("model", "unknown")
    print(f"Using model: {model}\n")

    # Discover stories
    stories = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])

    # Phase 1: Plan
    print(f"{'Story':<40} {'Words':>8} {'ChunkTok':>8} {'EstChunks':>9}")
    print("-" * 70)

    plan = []
    for story_dir in stories:
        name = story_dir.name
        merged_file = DATA_DIR / f"{name}_merged.txt"
        text = get_text_for_story(story_dir)
        total_words = len(text.split())

        if total_words < 100:
            print(f"{name:<40} {'SKIP':>8}  (too short: {total_words}w)")
            continue

        merged_file.write_text(text, encoding="utf-8")
        chunk_tokens = compute_chunk_tokens(total_words)
        est = max(1, total_words // max(1, int(chunk_tokens / WORDS_PER_TOKEN)))
        print(f"{name:<40} {total_words:>8} {chunk_tokens:>8} {est:>9}")
        plan.append((name, merged_file, chunk_tokens, total_words))

    print(f"\n{len(plan)} stories to analyze. Starting...\n")

    # Phase 2: Run PCMFG analysis on each story
    for i, (name, merged_file, chunk_tokens, total_words) in enumerate(plan):
        out_dir = OUTPUT_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        result_file = out_dir / "emotional_trajectory.json"

        # Skip if already has enough chunks
        if result_file.exists():
            with open(result_file) as f:
                existing = json.load(f)
            n = existing.get("metadata", {}).get("total_chunks", 0)
            if n >= MIN_CHUNKS_THRESHOLD:
                print(
                    f"[{i+1}/{len(plan)}] {name}: SKIP "
                    f"(already {n} chunks)"
                )
                continue

        # Write dynamic config
        cfg_path = make_config(chunk_tokens)

        print(
            f"[{i+1}/{len(plan)}] {name} "
            f"({total_words}w, chunk={chunk_tokens}tok)..."
        )

        result = subprocess.run(
            ["pcmfg", "analyze", str(merged_file),
             "-o", str(out_dir), "-c", str(cfg_path), "--no-plot"],
            capture_output=True, text=True, timeout=3600,
        )

        if result_file.exists():
            with open(result_file) as f:
                r = json.load(f)
            chunks = r.get("metadata", {}).get("total_chunks", 0)
            pairing = " & ".join(
                r.get("world_builder", {}).get("main_pairing", [])[:2]
            )
            print(f"  -> {chunks} chunks | {pairing}")
        else:
            stderr = result.stderr[-300:] if result.stderr else "no stderr"
            print(f"  FAILED: {stderr}")

        cfg_path.unlink(missing_ok=True)

    # Phase 3: Interesting section detection on all results
    print(f"\n{'='*70}")
    print("Running Interesting Section Detection on all results...")
    print(f"{'='*70}\n")

    for name, _, _, _ in plan:
        result_file = OUTPUT_DIR / name / "emotional_trajectory.json"
        if not result_file.exists():
            continue

        cmd = ["pcmfg", "interesting", str(result_file),
               "-o", str(OUTPUT_DIR / name)]
        subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        report_file = OUTPUT_DIR / name / "interesting_sections.json"
        if report_file.exists():
            with open(report_file) as f:
                r = json.load(f)
            pairing = " & ".join(r["main_pairing"][:2])
            print(
                f"  {name:<40} {pairing:<30} "
                f"{r['n_chunks']:>3} chunks, "
                f"{len(r['discords'])} discords, "
                f"{len(r['segments'])} segments"
            )

    # Cleanup merged files
    for f in DATA_DIR.glob("*_merged.txt"):
        f.unlink()

    print("\nDone! Results in output/")


if __name__ == "__main__":
    main()
