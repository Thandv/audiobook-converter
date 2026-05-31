"""`audiobook train ...` subcommands.

This module wires the dataset / training / inference helpers into Click.
The heavy deps (torch, coqui-tts, librosa) are imported lazily inside
the helpers, so importing this module itself stays cheap.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table


console = Console()


@click.group()
def train() -> None:
    """Fine-tune an emotion-aware TTS model and use it as a backend.

    Workflow:
      1. Get an emotion-labeled dataset (RAVDESS, ESD, or your own recordings)
      2. `audiobook train ingest` -> manifest.csv
      3. `audiobook train prepare` -> training-ready directory
      4. `audiobook train run` -> fine-tuned model directory
      5. `audiobook render --backend xtts --model-dir <model>`
    """


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@train.command()
@click.option(
    "--source",
    type=click.Choice(["ravdess", "esd", "custom"]),
    required=True,
    help="Dataset format to import.",
)
@click.option(
    "--path",
    "src_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the root of the dataset (or, for custom, an existing manifest.csv).",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to write the manifest.csv.",
)
def ingest(source: str, src_path: Path, out_path: Path) -> None:
    """Import a public dataset into our canonical manifest format."""
    from .dataset import ingest_esd, ingest_ravdess

    if source == "ravdess":
        n = ingest_ravdess(src_path, out_path)
        console.print(f"[green]Imported {n} RAVDESS rows -> {out_path}[/green]")
    elif source == "esd":
        n = ingest_esd(src_path, out_path)
        console.print(f"[green]Imported {n} ESD rows -> {out_path}[/green]")
    elif source == "custom":
        # Custom source: assume `src_path` IS a manifest, just validate + copy.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(src_path.read_bytes())
        console.print(f"[green]Copied custom manifest -> {out_path}[/green]")
    else:
        raise click.UsageError(f"Unknown source: {source}")

    # Validate immediately.
    from .dataset import validate_manifest
    issues = validate_manifest(out_path)
    if issues:
        console.print(f"[yellow]Manifest has {len(issues)} issue(s):[/yellow]")
        for i in issues[:20]:
            console.print(f"  - {i}")
        if len(issues) > 20:
            console.print(f"  ... ({len(issues) - 20} more)")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@train.command()
@click.option(
    "--manifest",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def validate(manifest: Path) -> None:
    """Validate a manifest.csv (checks paths, columns, emotion vocabulary)."""
    from .dataset import validate_manifest

    issues = validate_manifest(manifest)
    if not issues:
        console.print(f"[green]Manifest OK: {manifest}[/green]")
        return
    console.print(f"[yellow]{len(issues)} issue(s) found:[/yellow]")
    for i in issues:
        console.print(f"  - {i}")
    raise click.exceptions.Exit(code=1)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@train.command()
@click.option(
    "--manifest",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def stats(manifest: Path) -> None:
    """Show per-speaker / per-emotion counts and total duration."""
    from .dataset import dataset_stats

    s = dataset_stats(manifest)
    table = Table(title=f"Dataset: {manifest}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total clips", f"{s['total_rows']:,}")
    table.add_row("Total duration", str(s["total_duration_hms"]))
    console.print(table)

    emo_table = Table(title="Clips per emotion")
    emo_table.add_column("Emotion", style="cyan")
    emo_table.add_column("Clips", justify="right")
    for emo, n in sorted(s["per_emotion"].items(), key=lambda kv: -kv[1]):
        emo_table.add_row(emo, f"{n:,}")
    console.print(emo_table)

    sp_table = Table(title="Clips per speaker (top 20)")
    sp_table.add_column("Speaker", style="cyan")
    sp_table.add_column("Clips", justify="right")
    for sp, n in sorted(s["per_speaker"].items(), key=lambda kv: -kv[1])[:20]:
        sp_table.add_row(sp, f"{n:,}")
    console.print(sp_table)


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


@train.command()
@click.option(
    "--manifest",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Destination directory for the prepared (resampled + clipped) dataset.",
)
def prepare(manifest: Path, out_dir: Path) -> None:
    """Resample + clip audio to XTTS v2 requirements (22.05 kHz mono, <=11s)."""
    from .dataset import prepare_for_xtts

    result = prepare_for_xtts(manifest, out_dir)
    console.print(f"[green]Prepared dataset at: {result}[/green]")
    console.print("[dim]Next: audiobook train run --data {} --out <model_dir>[/dim]".format(result))


# ---------------------------------------------------------------------------
# run (fine-tune)
# ---------------------------------------------------------------------------


@train.command()
@click.option(
    "--data",
    "data_dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Prepared dataset directory (output of `train prepare`).",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Where to put checkpoints + reference clips.",
)
@click.option("--epochs", type=int, default=10)
@click.option("--batch-size", "batch_size", type=int, default=4)
@click.option("--lr", type=float, default=5e-6)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    default="auto",
)
def run(
    data_dir: Path, out_dir: Path,
    epochs: int, batch_size: int, lr: float, device: str,
) -> None:
    """Fine-tune XTTS v2 on the prepared dataset.

    Requires the training extras: `pip install -e ".[training]"`.

    Expect this to take hours-to-days depending on hardware. CPU training
    is not realistic for fine-tuning XTTS; use CUDA or MPS.
    """
    from .train import fine_tune_xtts

    console.print(f"[bold cyan]Fine-tuning XTTS v2[/bold cyan] (device={device}, epochs={epochs})")
    best = fine_tune_xtts(
        prepared_data_dir=data_dir,
        output_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    console.print(f"[green]Done. Best checkpoint: {best}[/green]")
    console.print(
        "[dim]Try it: audiobook render <manuscript.md> --backend xtts "
        f"--model-dir {out_dir}[/dim]"
    )


# ---------------------------------------------------------------------------
# test (sample-render from a trained model)
# ---------------------------------------------------------------------------


@train.command()
@click.option(
    "--model",
    "model_dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("--speaker", required=True, help="Speaker name to use for the sample.")
@click.option("--emotion", default="neutral", help="Emotion label (e.g. angry, sad, whispered).")
@click.option("--text", required=True, help="Text to synthesize.")
@click.option(
    "--out",
    "out_wav",
    type=click.Path(path_type=Path),
    default=Path("test_sample.wav"),
)
@click.option("--speed", type=float, default=1.0)
def test(
    model_dir: Path, speaker: str, emotion: str, text: str,
    out_wav: Path, speed: float,
) -> None:
    """Render a single test sample from a fine-tuned model."""
    import soundfile as sf
    from .infer import FineTunedXTTSSynth, TARGET_SAMPLE_RATE

    synth = FineTunedXTTSSynth(model_dir)
    if speaker not in synth.available_speakers():
        console.print(
            f"[red]Unknown speaker '{speaker}'. Available: "
            f"{', '.join(synth.available_speakers())}[/red]"
        )
        raise click.exceptions.Exit(code=1)
    available_emos = synth.available_emotions(speaker)
    if emotion not in available_emos:
        console.print(
            f"[yellow]Emotion '{emotion}' not in model for speaker '{speaker}'. "
            f"Available: {', '.join(available_emos)}. Falling back at inference time.[/yellow]"
        )
    audio = synth.synthesize(text, speaker=speaker, emotion=emotion, speed=speed)
    sf.write(str(out_wav), audio, TARGET_SAMPLE_RATE)
    duration = audio.size / TARGET_SAMPLE_RATE
    console.print(f"[green]Wrote {out_wav} ({duration:.1f}s)[/green]")


# ---------------------------------------------------------------------------
# inspect (introspect a trained model)
# ---------------------------------------------------------------------------


@train.command()
@click.option(
    "--model",
    "model_dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def inspect(model_dir: Path) -> None:
    """List speakers and emotions available in a fine-tuned model."""
    manifest_path = model_dir / "inference.json"
    if not manifest_path.exists():
        console.print(f"[red]No inference.json in {model_dir} — was this dir produced by `train run`?[/red]")
        raise click.exceptions.Exit(code=1)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    speakers = data.get("speakers", {})
    table = Table(title=f"Model: {model_dir.name}")
    table.add_column("Speaker", style="cyan")
    table.add_column("Emotions available")
    for sp, emos in sorted(speakers.items()):
        table.add_row(sp, ", ".join(emos) if emos else "—")
    console.print(table)
