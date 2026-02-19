"""Novel loader utility for PCMFG.

Loads novels from structured directories containing chapter text files
and merges them into a single text with proper chapter markers.
"""

import re
from pathlib import Path


def load_novel_from_directory(
    novel_dir: str | Path,
    output_file: str | Path | None = None,
    chapter_prefix: str = "Chapter",
) -> str:
    """Load a novel from a directory structure and merge into single text.

    Directory structure expected:
        novel_dir/
            collection_1/
                001_Chapter 1.txt
                002_Chapter 2.txt
                ...
            collection_2/
                001_Chapter 10.txt
                ...

    Args:
        novel_dir: Path to the novel directory containing collection folders.
        output_file: Optional path to save the merged novel.
        chapter_prefix: Prefix to use for chapter markers (default: "Chapter").

    Returns:
        Merged novel text with chapter markers.
    """
    novel_dir = Path(novel_dir)

    if not novel_dir.exists():
        raise FileNotFoundError(f"Novel directory not found: {novel_dir}")

    # Find all collection folders (sorted)
    collections = sorted([d for d in novel_dir.iterdir() if d.is_dir()])

    if not collections:
        raise ValueError(f"No collection folders found in: {novel_dir}")

    all_chapters: list[tuple[int, str, str]] = []  # (sort_key, chapter_title, content)

    for collection_dir in collections:
        # Find all chapter files in this collection
        chapter_files = sorted(collection_dir.glob("*.txt"))

        for chapter_file in chapter_files:
            chapter_title, chapter_num = _parse_chapter_filename(
                chapter_file.name, chapter_prefix
            )

            with open(chapter_file, encoding="utf-8") as f:
                content = f.read().strip()

            # Use chapter number for sorting, or file index if not found
            sort_key = chapter_num if chapter_num is not None else len(all_chapters)
            all_chapters.append((sort_key, chapter_title, content))

    # Sort by chapter number
    all_chapters.sort(key=lambda x: x[0])

    # Build merged novel text
    novel_parts = []
    for sort_key, chapter_title, content in all_chapters:
        novel_parts.append(f"Chapter {sort_key}\n\n{content}")

    merged_text = "\n\n".join(novel_parts)

    # Save to file if output path provided
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(merged_text)

    return merged_text


def _parse_chapter_filename(
    filename: str,
    chapter_prefix: str = "Chapter",
) -> tuple[str, int | None]:
    """Parse chapter title and number from filename.

    Args:
        filename: The chapter filename (e.g., "001_Chapter 41.txt").
        chapter_prefix: The expected chapter prefix.

    Returns:
        Tuple of (chapter_title, chapter_number or None).
    """
    # Remove .txt extension
    name = Path(filename).stem

    # Pattern 1: "001_Chapter 41" -> extract "Chapter 41" and number 41
    match = re.search(r"Chapter\s*(\d+)", name, re.IGNORECASE)
    if match:
        chapter_num = int(match.group(1))
        return f"Chapter {chapter_num}", chapter_num

    # Pattern 2: Side stories "ss-5- Side Story 5" -> "Side Story 5"
    match = re.search(r"Side Story\s*(\d+)", name, re.IGNORECASE)
    if match:
        side_num = int(match.group(1))
        # Use high number range for side stories (e.g., 1000+)
        return f"Side Story {side_num}", 1000 + side_num

    # Pattern 3: Just extract any number from filename
    match = re.search(r"(\d+)", name)
    if match:
        chapter_num = int(match.group(1))
        return name, chapter_num

    # Fallback: use filename as title
    return name, None


def get_novel_info(novel_dir: str | Path) -> dict:
    """Get information about a novel directory.

    Args:
        novel_dir: Path to the novel directory.

    Returns:
        Dictionary with novel information.
    """
    novel_dir = Path(novel_dir)

    collections = sorted([d for d in novel_dir.iterdir() if d.is_dir()])

    total_chapters = 0
    chapter_range = {"start": None, "end": None}
    has_side_stories = False

    for collection_dir in collections:
        chapter_files = list(collection_dir.glob("*.txt"))
        total_chapters += len(chapter_files)

        for chapter_file in chapter_files:
            _, chapter_num = _parse_chapter_filename(chapter_file.name)
            if chapter_num is not None and chapter_num < 1000:
                if (
                    chapter_range["start"] is None
                    or chapter_num < chapter_range["start"]
                ):
                    chapter_range["start"] = chapter_num
                if chapter_range["end"] is None or chapter_num > chapter_range["end"]:
                    chapter_range["end"] = chapter_num
            elif chapter_num is not None and chapter_num >= 1000:
                has_side_stories = True

    return {
        "path": str(novel_dir),
        "collections": [c.name for c in collections],
        "total_files": total_chapters,
        "chapter_range": chapter_range,
        "has_side_stories": has_side_stories,
    }


class NovelLoader:
    """Class-based novel loader for more control over loading process."""

    def __init__(
        self,
        novel_dir: str | Path,
        chapter_prefix: str = "Chapter",
    ) -> None:
        """Initialize novel loader.

        Args:
            novel_dir: Path to the novel directory.
            chapter_prefix: Prefix for chapter markers.
        """
        self.novel_dir = Path(novel_dir)
        self.chapter_prefix = chapter_prefix
        self._info: dict | None = None

    @property
    def info(self) -> dict:
        """Get novel information (cached)."""
        if self._info is None:
            self._info = get_novel_info(self.novel_dir)
        return self._info

    def load(self, output_file: str | Path | None = None) -> str:
        """Load the novel and optionally save to file.

        Args:
            output_file: Optional output file path.

        Returns:
            Merged novel text.
        """
        return load_novel_from_directory(
            self.novel_dir,
            output_file=output_file,
            chapter_prefix=self.chapter_prefix,
        )

    def load_chapters(self) -> list[dict]:
        """Load novel as list of chapter dictionaries.

        Returns:
            List of dicts with 'chapter_num', 'title', 'content' keys.
        """
        collections = sorted([d for d in self.novel_dir.iterdir() if d.is_dir()])

        all_chapters: list[tuple[int, dict]] = []

        for collection_dir in collections:
            chapter_files = sorted(collection_dir.glob("*.txt"))

            for chapter_file in chapter_files:
                title, chapter_num = _parse_chapter_filename(
                    chapter_file.name, self.chapter_prefix
                )

                with open(chapter_file, encoding="utf-8") as f:
                    content = f.read().strip()

                sort_key = chapter_num if chapter_num is not None else len(all_chapters)
                all_chapters.append(
                    (
                        sort_key,
                        {
                            "chapter_num": chapter_num,
                            "title": title,
                            "content": content,
                            "source_file": str(chapter_file),
                        },
                    )
                )

        all_chapters.sort(key=lambda x: x[0])
        return [chapter for _, chapter in all_chapters]
