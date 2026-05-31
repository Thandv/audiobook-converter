"""`audiobook voices ...` subcommand group — manage the voice library.

Subcommands:
  list                  Show every clip and its metadata.
  show <character>      Coverage matrix for one character.
  import                Import a single audio file into the library.
  import-ravdess        Bulk-import a RAVDESS dataset as stock emotion clips.
  record                Record a clip interactively (uses your mic).
  delete                Remove a clip.
  validate              Check all clips for length / sample-rate issues.
  templates             Print sample sentences the user can read aloud
                        when recording new emotional clips.
  coverage              Show missing (character, emotion) slots vs. a cast.
"""
from __future__ import annotations

import re
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .emotion import EMOTIONS
from .voice_library import VoiceLibrary, ClipInfo


console = Console()


# Map RAVDESS actor IDs to default character roles. Customize after import.
RAVDESS_EMOTION_CODE = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgusted",
    "08": "surprised",
}


# Template sentences for the user to read aloud when self-recording.
# Designed to evoke the target emotion naturally.
RECORDING_TEMPLATES: dict[str, list[str]] = {
    "neutral": [
        "The stones held through the night. We walked the perimeter at dawn, the way we always have.",
        "Tell me what you saw, in your own words. Begin at the beginning.",
    ],
    "calm": [
        "Breathe slowly. There is nothing in this room that cannot wait for the next breath.",
        "Listen to the river. It has been here longer than any of us, and it asks nothing.",
    ],
    "happy": [
        "She caught it on the first try — caught it cleanly — and her whole face lit up like a window in winter.",
        "We did it. We actually did it. Look at this, look at this — it works!",
    ],
    "sad": [
        "He didn't say a word. He just sat down, and the room got smaller, and we knew.",
        "I keep listening for her voice in the corridor, and there's nothing. Just the wind.",
    ],
    "angry": [
        "You knew. You knew, and you said nothing, and now eighty-three people are dead.",
        "Get out. Get out of my house. I never want to see your face here again.",
    ],
    "fearful": [
        "Something is in the dark behind us. I can hear it breathing. Don't turn around.",
        "Please. Please don't. I'm begging you, please.",
    ],
    "surprised": [
        "What — wait, what? You're telling me she was here? Here, the whole time?",
        "Oh — oh god — I didn't see you there. You almost stopped my heart.",
    ],
    "disgusted": [
        "Get that away from me. The smell — I can't, I can't be near it.",
        "How could you. How could you even think to do something like that.",
    ],
    "whispered": [
        "Don't speak. Don't move. They're right outside the door, and they're listening.",
        "Come closer. I have to tell you something, and the walls have ears tonight.",
    ],
    "excited": [
        "We have to go, we have to go now! Come on, come on, before they close the gate!",
        "Did you hear what she said? Did you HEAR her? This changes everything!",
    ],
}


def _library_root() -> Path:
    """Default library root: <project>/voices/."""
    return Path(__file__).resolve().parents[2] / "voices"


@click.group()
def voices() -> None:
    """Manage the emotion-tagged reference clip library.

    The library is a folder tree of:
      voices/<character>/<emotion>.wav

    These clips are the prompts that voice-cloning backends (xtts, chatterbox)
    use to copy a voice and its emotional prosody.
    """


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@voices.command("list")
@click.option("--library", "library_path", type=click.Path(path_type=Path), default=None)
@click.option("--character", default=None, help="Filter to one character.")
def list_clips(library_path: Path | None, character: str | None) -> None:
    """List every clip with duration, sample rate, channels, and issues."""
    lib = VoiceLibrary(library_path or _library_root())
    infos = lib.list_clips(character)
    if not infos:
        console.print(f"[yellow]No clips found in {lib.root}[/yellow]")
        return
    table = Table(title=f"Library: {lib.root}")
    table.add_column("Character", style="cyan")
    table.add_column("Emotion")
    table.add_column("Duration", justify="right")
    table.add_column("SR", justify="right")
    table.add_column("Ch", justify="right")
    table.add_column("Issues")
    for c in infos:
        issues = "; ".join(c.issues) if c.issues else "[green]ok[/green]"
        table.add_row(
            c.character, c.emotion, f"{c.duration_seconds:.1f}s",
            str(c.sample_rate), str(c.channels), issues,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# show <character>
# ---------------------------------------------------------------------------


@voices.command("show")
@click.argument("character")
@click.option("--library", "library_path", type=click.Path(path_type=Path), default=None)
def show_character(character: str, library_path: Path | None) -> None:
    """Show emotion coverage for a single character."""
    lib = VoiceLibrary(library_path or _library_root())
    table = Table(title=f"{character}")
    table.add_column("Emotion", style="cyan")
    table.add_column("Status")
    table.add_column("Path")
    for emo in EMOTIONS:
        clip = lib.find_clip(character, emo)
        if clip:
            status = "[green]present[/green]"
            path = str(clip.relative_to(lib.root.parent)) if clip.is_absolute() else str(clip)
        else:
            status = "[red]missing[/red]"
            path = ""
        table.add_row(emo, status, path)
    console.print(table)


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


@voices.command("coverage")
@click.option("--library", "library_path", type=click.Path(path_type=Path), default=None)
@click.option(
    "--voices-yaml", "voices_yaml",
    type=click.Path(exists=True, path_type=Path), default=None,
    help="Read the cast list from voices.yaml to know which characters are expected.",
)
@click.option(
    "--emotions",
    default="neutral,angry,sad,whispered,excited",
    help="Comma-separated emotions to require for each character.",
)
def coverage(library_path: Path | None, voices_yaml: Path | None, emotions: str) -> None:
    """Show which (character, emotion) slots are missing."""
    lib = VoiceLibrary(library_path or _library_root())
    voices_yaml = voices_yaml or (Path(__file__).resolve().parents[2] / "config" / "voices.yaml")

    from .synth import load_voice_cast
    cast = load_voice_cast(voices_yaml)
    characters = ["narrator"] + sorted(cast.cast.keys())
    required = [e.strip() for e in emotions.split(",") if e.strip()]

    missing = lib.missing(characters, required)
    if not missing:
        console.print("[green]All required (character, emotion) slots are present.[/green]")
        return

    table = Table(title=f"Missing: {len(missing)} slots")
    table.add_column("Character", style="cyan")
    table.add_column("Emotion")
    for ch, emo in missing:
        table.add_row(ch, emo)
    console.print(table)
    console.print(
        f"\n[dim]Tip: `audiobook voices import --character X --emotion Y --file path.wav`[/dim]"
    )


# ---------------------------------------------------------------------------
# import (single file)
# ---------------------------------------------------------------------------


@voices.command("import")
@click.option("--character", required=True)
@click.option("--emotion", required=True, type=click.Choice(list(EMOTIONS)))
@click.option("--file", "src_file", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--library", "library_path", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite", is_flag=True)
@click.option("--no-normalize", is_flag=True,
              help="Copy as-is (default: resample to 22050 Hz mono WAV).")
def import_single(
    character: str, emotion: str, src_file: Path,
    library_path: Path | None, overwrite: bool, no_normalize: bool,
) -> None:
    """Import one audio clip into the library."""
    lib = VoiceLibrary(library_path or _library_root())
    try:
        dst = lib.import_clip(
            src_file, character, emotion,
            overwrite=overwrite, normalize=not no_normalize,
        )
    except FileExistsError as e:
        console.print(f"[red]{e}[/red]")
        raise click.exceptions.Exit(code=1)
    except (ValueError, RuntimeError) as e:
        console.print(f"[red]{e}[/red]")
        raise click.exceptions.Exit(code=1)
    console.print(f"[green]Imported -> {dst}[/green]")


# ---------------------------------------------------------------------------
# import-ravdess (bulk)
# ---------------------------------------------------------------------------


_RAVDESS_RE = re.compile(
    r"^(?P<mod>\d{2})-(?P<vc>\d{2})-(?P<emo>\d{2})-"
    r"(?P<intensity>\d{2})-(?P<stmt>\d{2})-(?P<rep>\d{2})-(?P<actor>\d{2})\.wav$"
)


@voices.command("import-ravdess")
@click.option("--path", "ravdess_root", type=click.Path(exists=True, path_type=Path), required=True,
              help="Path to the RAVDESS Speech dataset root.")
@click.option("--library", "library_path", type=click.Path(path_type=Path), default=None)
@click.option(
    "--actors", default="auto",
    help="Comma-separated actor IDs (e.g. 01,02,15). 'auto' uses two actors "
         "(one M, one F) which are mapped to 'narrator' and your story's main female. "
         "Pass explicit IDs to map specific actors to specific characters via --map.",
)
@click.option(
    "--map", "char_map", default=None,
    help="character=actor_id mapping (e.g. 'Gael=01,Sera=02,narrator=11'). "
         "Each actor's full 8-emotion set becomes that character's library.",
)
@click.option("--overwrite", is_flag=True)
def import_ravdess(
    ravdess_root: Path, library_path: Path | None,
    actors: str, char_map: str | None, overwrite: bool,
) -> None:
    """Bulk-import a RAVDESS dataset as stock emotion templates.

    Each actor in RAVDESS records 8 emotions x 2 sentences x 2 repetitions.
    We pick the best 'normal intensity' repetition per emotion and copy it
    into the library under the mapped character name.

    By default, two actors (one male, one female) are imported as 'narrator'
    and '_other'. Use --map to assign specific actors to your characters.
    """
    lib = VoiceLibrary(library_path or _library_root())

    # Build the character->actor mapping.
    mapping: dict[str, str] = {}
    if char_map:
        for pair in char_map.split(","):
            if "=" not in pair:
                console.print(f"[red]Invalid mapping '{pair}'; expected 'character=actor_id'[/red]")
                raise click.exceptions.Exit(code=1)
            ch, actor = pair.split("=", 1)
            mapping[ch.strip()] = actor.strip().zfill(2)
    else:
        if actors == "auto":
            # Default: actor 11 (M, calm baritone) -> narrator
            #          actor 02 (F, expressive)   -> _woman fallback
            mapping = {"narrator": "11", "_woman": "02", "_man": "11"}
        else:
            for i, actor in enumerate(actors.split(",")):
                actor = actor.strip().zfill(2)
                mapping[f"actor_{actor}"] = actor

    # For each character/actor, find the best clip per emotion.
    n_imported = 0
    for character, actor_id in mapping.items():
        # Look for normal-intensity (01) clips first, fall back to strong (02).
        for emo_code, emo_name in RAVDESS_EMOTION_CODE.items():
            best: Path | None = None
            for intensity in ("01", "02"):
                candidates = sorted(
                    ravdess_root.rglob(f"03-01-{emo_code}-{intensity}-*-*-{actor_id}.wav")
                )
                if candidates:
                    best = candidates[0]
                    break
            if best is None:
                console.print(
                    f"[yellow]No RAVDESS clip for actor {actor_id}, emotion {emo_name}[/yellow]"
                )
                continue
            try:
                lib.import_clip(best, character, emo_name, overwrite=overwrite, normalize=True)
                n_imported += 1
            except FileExistsError:
                console.print(f"[dim]Skipping {character}/{emo_name} (exists; pass --overwrite)[/dim]")
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]Failed {character}/{emo_name}: {e}[/red]")

    console.print(f"\n[green]Imported {n_imported} RAVDESS clips into {lib.root}[/green]")
    console.print(
        f"[dim]Try: audiobook voices list --library {lib.root}[/dim]"
    )


# ---------------------------------------------------------------------------
# record (mic)
# ---------------------------------------------------------------------------


@voices.command("record")
@click.option("--character", required=True)
@click.option("--emotion", required=True, type=click.Choice(list(EMOTIONS)))
@click.option("--seconds", type=int, default=15, help="Recording length.")
@click.option("--library", "library_path", type=click.Path(path_type=Path), default=None)
@click.option("--overwrite", is_flag=True)
def record(
    character: str, emotion: str, seconds: int,
    library_path: Path | None, overwrite: bool,
) -> None:
    """Record a reference clip from your microphone.

    Requires `pip install sounddevice` (added when you install the
    'training' extras).
    """
    try:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
    except ImportError as e:
        console.print(f"[red]Recording requires sounddevice. Install: pip install sounddevice[/red]")
        console.print(f"[dim]{e}[/dim]")
        raise click.exceptions.Exit(code=1)

    lib = VoiceLibrary(library_path or _library_root())
    if lib.has_clip(character, emotion) and not overwrite:
        console.print(
            f"[red]Clip already exists for {character}/{emotion}. "
            f"Pass --overwrite to replace.[/red]"
        )
        raise click.exceptions.Exit(code=1)

    # Show a prompt for the user to read aloud.
    templates = RECORDING_TEMPLATES.get(emotion, [])
    if templates:
        console.print(f"\n[bold cyan]Read aloud (with {emotion} delivery):[/bold cyan]")
        for t in templates[:2]:
            console.print(f"  • [italic]{t}[/italic]")

    console.print(
        f"\nRecording {seconds}s in 3... ", end="", style="bold"
    )
    import time
    for n in (3, 2, 1):
        console.print(f"{n} ", end="")
        time.sleep(1)
    console.print("[bold red]REC[/bold red]")

    sr = 22050
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    console.print("[dim]Done.[/dim]")

    char_dir = lib.character_dir(character)
    char_dir.mkdir(parents=True, exist_ok=True)
    dst = char_dir / f"{emotion}.wav"
    sf.write(str(dst), audio, sr, subtype="PCM_16")
    console.print(f"[green]Saved -> {dst}[/green]")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@voices.command("delete")
@click.option("--character", required=True)
@click.option("--emotion", required=True, type=click.Choice(list(EMOTIONS)))
@click.option("--library", "library_path", type=click.Path(path_type=Path), default=None)
def delete_clip(character: str, emotion: str, library_path: Path | None) -> None:
    """Remove a single (character, emotion) clip from the library."""
    lib = VoiceLibrary(library_path or _library_root())
    if lib.delete_clip(character, emotion):
        console.print(f"[green]Deleted {character}/{emotion}[/green]")
    else:
        console.print(f"[yellow]No clip found for {character}/{emotion}[/yellow]")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@voices.command("validate")
@click.option("--library", "library_path", type=click.Path(path_type=Path), default=None)
def validate(library_path: Path | None) -> None:
    """Check all clips for duration / sample rate / channel issues."""
    lib = VoiceLibrary(library_path or _library_root())
    infos = lib.list_clips()
    if not infos:
        console.print(f"[yellow]Library is empty: {lib.root}[/yellow]")
        return
    bad = [c for c in infos if not c.ok]
    if not bad:
        console.print(f"[green]All {len(infos)} clip(s) look fine.[/green]")
        return
    table = Table(title=f"Clips with issues ({len(bad)} of {len(infos)})")
    table.add_column("Character", style="cyan")
    table.add_column("Emotion")
    table.add_column("Issue")
    for c in bad:
        for issue in c.issues:
            table.add_row(c.character, c.emotion, issue)
    console.print(table)


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------


@voices.command("templates")
@click.option("--emotion", type=click.Choice(list(EMOTIONS)), default=None)
def templates(emotion: str | None) -> None:
    """Print sample sentences to read when self-recording emotional clips."""
    emos = [emotion] if emotion else list(EMOTIONS)
    for emo in emos:
        lines = RECORDING_TEMPLATES.get(emo, [])
        if not lines:
            continue
        console.print(f"\n[bold cyan]{emo}[/bold cyan]")
        for ln in lines:
            console.print(f"  • [italic]{ln}[/italic]")
    console.print(
        "\n[dim]Aim for 10–15 seconds per recording. Speak with genuine emotion.[/dim]"
    )
