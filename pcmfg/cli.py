"""Command-line interface for PCMFG.

Provides the `pcmfg` command with subcommands:
- `analyze`: Analyze a text file
- `version`: Show version information
"""

import json
import logging
import sys
from pathlib import Path
from typing import Literal

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from pcmfg import __version__
from pcmfg.analyzer import PCMFGAnalyzer
from pcmfg.config import Config, load_config, merge_cli_overrides
from pcmfg.models.schemas import AnalysisResult
from pcmfg.visualization.plotter import EmotionPlotter

# Load environment variables from .env file
load_dotenv()

console = Console()


def setup_logging(debug: bool = False) -> None:
    """Set up logging configuration.

    Args:
        debug: If True, enable debug logging.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """PCMFG - Please Care My Feeling Graph

    A computational romance narrative mining system.
    """
    setup_logging(debug)
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="./output",
    help="Output directory for results",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to config file",
)
@click.option(
    "-p",
    "--provider",
    type=click.Choice(["openai", "anthropic"]),
    default=None,
    help="LLM provider to use",
)
@click.option(
    "-m",
    "--model",
    type=str,
    default=None,
    help="Model name to use",
)
@click.option(
    "-f",
    "--format",
    type=click.Choice(["json", "csv", "both"]),
    default="json",
    help="Output format for data",
)
@click.option(
    "--no-plot",
    is_flag=True,
    help="Skip generating visualization plot",
)
@click.option(
    "--stats",
    is_flag=True,
    help="Include statistical analysis",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    input_file: str,
    output: str,
    config: str | None,
    provider: Literal["openai", "anthropic"] | None,
    model: str | None,
    format: Literal["json", "csv", "both"],
    no_plot: bool,
    stats: bool,
) -> None:
    """Analyze a romantic narrative text file.

    INPUT_FILE: Path to the text file to analyze.
    """
    debug = ctx.obj.get("debug", False)

    # Load configuration
    cfg = load_config(config)

    # Apply CLI overrides
    if provider:
        cfg = merge_cli_overrides(cfg, llm__provider=provider)
    if model:
        cfg = merge_cli_overrides(cfg, llm__model=model)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_file)
    console.print(f"[bold blue]Analyzing:[/] {input_path.name}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Running PCMFG analysis...", total=None)

            # Create analyzer
            analyzer = PCMFGAnalyzer(config=cfg)

            # Run analysis
            result = analyzer.analyze_file(input_path)

        # Display summary
        _display_summary(result)

        # Export results
        _export_results(result, output_dir, format, cfg, no_plot)

        # Generate stats if requested
        if stats:
            _generate_stats_report(result, output_dir)

        console.print(f"\n[bold green]Done![/] Results saved to: {output_dir}")

    except ValueError as e:
        console.print(f"[bold red]Configuration error:[/] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Analysis failed:[/] {e}")
        if debug:
            console.print_exception()
        sys.exit(1)


def _display_summary(result: AnalysisResult) -> None:
    """Display analysis summary in a table."""
    table = Table(title="Analysis Summary")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Source", result.metadata.source or "Unknown")
    table.add_row("Total Chunks", str(result.metadata.total_chunks))
    table.add_row("Model", result.metadata.model)
    table.add_row(
        "Main Pairing",
        " & ".join(result.world_builder.main_pairing[:2]),
    )

    console.print(table)

    # Display axis averages
    if result.axes.intimacy:
        axis_table = Table(title="Average Axis Values")
        axis_table.add_column("Axis", style="cyan")
        axis_table.add_column("Average", style="green")
        axis_table.add_column("Min", style="yellow")
        axis_table.add_column("Max", style="red")

        for name in ["intimacy", "passion", "hostility", "anxiety"]:
            values = getattr(result.axes, name)
            if values:
                avg = sum(values) / len(values)
                axis_table.add_row(
                    name.capitalize(),
                    f"{avg:.2f}",
                    f"{min(values):.2f}",
                    f"{max(values):.2f}",
                )

        console.print(axis_table)


def _export_results(
    result: AnalysisResult,
    output_dir: Path,
    format: str,
    config: Config,
    no_plot: bool,
) -> None:
    """Export analysis results to files."""
    # Export JSON
    if format in ["json", "both"]:
        json_path = output_dir / "emotional_trajectory.json"
        with open(json_path, "w", encoding="utf-8") as f:
            # Convert to dict and handle datetime serialization
            data = result.model_dump(mode="json")
            json.dump(data, f, indent=2, default=str)
        console.print(f"  [dim]JSON:[/] {json_path}")

    # Export CSV
    if format in ["csv", "both"]:
        csv_path = output_dir / "emotional_trajectory.csv"
        plotter = EmotionPlotter()
        plotter.export_data(result.axes, csv_path, format="csv")
        console.print(f"  [dim]CSV:[/] {csv_path}")

    # Generate plot
    if not no_plot and "png" in config.output.formats:
        plot_path = output_dir / "emotional_trajectory.png"
        plotter = EmotionPlotter(dpi=config.output.dpi)
        plotter.plot_axes(
            result.axes,
            plot_path,
            title=f"Emotional Trajectory: {result.metadata.source}",
        )
        console.print(f"  [dim]Plot:[/] {plot_path}")


def _generate_stats_report(result: AnalysisResult, output_dir: Path) -> None:
    """Generate a statistical analysis report."""
    import numpy as np

    report_path = output_dir / "analysis_report.md"

    lines = [
        "# PCMFG Analysis Report",
        "",
        f"**Source:** {result.metadata.source or 'Unknown'}",
        f"**Date:** {result.metadata.analysis_date}",
        f"**Model:** {result.metadata.model}",
        "",
        "## Main Pairing",
        "",
        f"- {' & '.join(result.world_builder.main_pairing[:2])}",
        "",
        "## World Guidelines",
        "",
    ]

    for guideline in result.world_builder.world_guidelines:
        lines.append(f"- {guideline}")

    lines.extend(
        [
            "",
            "## Axis Statistics",
            "",
            "| Axis | Mean | Std Dev | Min | Max |",
            "|------|------|---------|-----|-----|",
        ]
    )

    for name in ["intimacy", "passion", "hostility", "anxiety"]:
        values = getattr(result.axes, name)
        if values:
            arr = np.array(values)
            lines.append(
                f"| {name.capitalize()} | {arr.mean():.2f} | {arr.std():.2f} | "
                f"{arr.min():.2f} | {arr.max():.2f} |"
            )

    lines.extend(
        [
            "",
            "## Trend Analysis",
            "",
        ]
    )

    # Simple trend analysis
    for name in ["intimacy", "passion", "hostility", "anxiety"]:
        values = getattr(result.axes, name)
        if values and len(values) >= 2:
            # Linear regression for trend
            x = np.arange(len(values))
            y = np.array(values)
            slope = np.polyfit(x, y, 1)[0]
            if slope > 0.1:
                trend = "increasing"
            elif slope < -0.1:
                trend = "decreasing"
            else:
                trend = "stable"
            lines.append(f"- **{name.capitalize()}**: {trend} (slope: {slope:.3f})")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"  [dim]Report:[/] {report_path}")


@cli.command()
def version() -> None:
    """Show PCMFG version information."""
    console.print(f"PCMFG version {__version__}")
    console.print("Please Care My Feeling Graph")
    console.print("A computational romance narrative mining system")


@cli.command()
@click.argument("novel_dir", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path for merged novel text",
)
@click.option(
    "--info",
    is_flag=True,
    help="Show novel information without loading",
)
def load(novel_dir: str, output: str | None, info: bool) -> None:
    """Load a novel from a directory structure.

    NOVEL_DIR: Path to the novel directory containing collection folders.
    """
    from pcmfg.utils.novel_loader import NovelLoader, get_novel_info

    novel_path = Path(novel_dir)

    if info:
        # Show novel info
        novel_info = get_novel_info(novel_path)
        table = Table(title="Novel Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Path", novel_info["path"])
        table.add_row("Collections", ", ".join(novel_info["collections"]))
        table.add_row("Total Files", str(novel_info["total_files"]))
        if novel_info["chapter_range"]["start"]:
            table.add_row(
                "Chapter Range",
                f"{novel_info['chapter_range']['start']} - {novel_info['chapter_range']['end']}",
            )
        table.add_row(
            "Has Side Stories", "Yes" if novel_info["has_side_stories"] else "No"
        )

        console.print(table)
        return

    # Load the novel
    loader = NovelLoader(novel_path)

    if output:
        output_path = Path(output)
    else:
        # Default output location
        output_path = Path("./output") / f"{novel_path.name}_merged.txt"

    console.print(f"[bold blue]Loading novel from:[/] {novel_path}")

    try:
        merged_text = loader.load(output_path)
        console.print(f"[bold green]Success![/] Merged novel saved to: {output_path}")
        console.print(f"  [dim]Total characters:[/] {len(merged_text):,}")
        console.print(f"  [dim]Total chapters:[/] {len(loader.load_chapters())}")
    except Exception as e:
        console.print(f"[bold red]Error loading novel:[/] {e}")
        raise


@cli.command()
@click.argument("novel_dir", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="./output",
    help="Output directory for analysis results",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to config file",
)
@click.option(
    "--start",
    type=int,
    default=None,
    help="Start chapter number",
)
@click.option(
    "--end",
    type=int,
    default=None,
    help="End chapter number",
)
@click.option(
    "--save-merged",
    is_flag=True,
    default=True,
    help="Save merged novel text",
)
@click.pass_context
def analyze_novel(
    ctx: click.Context,
    novel_dir: str,
    output: str,
    config: str | None,
    start: int | None,
    end: int | None,
    save_merged: bool,
) -> None:
    """Load and analyze a novel from a directory structure.

    NOVEL_DIR: Path to the novel directory containing collection folders.
    """
    from pcmfg.utils.novel_loader import NovelLoader

    debug = ctx.obj.get("debug", False)
    novel_path = Path(novel_dir)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    cfg = load_config(config)

    console.print(f"[bold blue]Loading novel:[/] {novel_path.name}")

    try:
        loader = NovelLoader(novel_path)
        chapters = loader.load_chapters()

        # Filter chapters by range
        if start is not None:
            chapters = [
                c
                for c in chapters
                if c["chapter_num"] is not None and c["chapter_num"] >= start
            ]
        if end is not None:
            chapters = [
                c
                for c in chapters
                if c["chapter_num"] is not None and c["chapter_num"] <= end
            ]

        # Filter out side stories if needed (chapters >= 1000)
        chapters = [
            c for c in chapters if c["chapter_num"] is None or c["chapter_num"] < 1000
        ]

        console.print(f"  [dim]Chapters to analyze:[/] {len(chapters)}")

        # Build merged text
        merged_text = "\n\n".join(
            f"Chapter {c['chapter_num']}\n\n{c['content']}" for c in chapters
        )

        # Save merged text if requested
        if save_merged:
            merged_path = output_dir / f"{novel_path.name}_merged.txt"
            with open(merged_path, "w", encoding="utf-8") as f:
                f.write(merged_text)
            console.print(f"  [dim]Merged text:[/] {merged_path}")

        # Run analysis
        console.print(f"[bold blue]Analyzing novel...[/]")

        analyzer = PCMFGAnalyzer(config=cfg)
        result = analyzer.analyze(merged_text, source=novel_path.name)

        # Display and export results
        _display_summary(result)
        _export_results(result, output_dir, "json", cfg, False)
        _generate_stats_report(result, output_dir)

        console.print(f"\n[bold green]Done![/] Results saved to: {output_dir}")

    except Exception as e:
        console.print(f"[bold red]Analysis failed:[/] {e}")
        if debug:
            console.print_exception()
        raise


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
