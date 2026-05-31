# Audiobook Converter

Convert a fiction manuscript in Markdown into a professional-quality M4B
audiobook with chapter markers and realistic voices.

- **Default backend: [Kokoro TTS](https://github.com/hexgrad/kokoro)** —
  free, local, 50+ voices, no GPU required.
- **Optional backend: fine-tuned XTTS v2** — train your own emotion-aware
  voices on labeled audio data and use them in the pipeline.
- **Modes**: single-narrator (traditional) or multi-voice (different voice
  per character via rule-based dialogue attribution).
- **Emotion**: dialogue tags ("whispered", "shouted", "growled"…) are
  auto-detected and routed through per-emotion voice overrides.
- **Interactive UX**: run `audiobook` with no arguments and you get a
  guided prompt-driven walkthrough.
- **Two install paths**: `pipx install` for daily use, or
  `make binary` for a standalone executable you can hand to non-Python
  users.

## Quick start

```bash
# One-time system deps (macOS)
brew install espeak-ng ffmpeg pipx

# Global install of the `audiobook` command
pipx install --editable .

# Interactive walkthrough
audiobook
```

That's it. The first run downloads ~330 MB of Kokoro model weights to
`~/.cache/huggingface/`. After that everything is offline.

## Install options

| Method                               | Use when                                                  |
|--------------------------------------|-----------------------------------------------------------|
| `pipx install --editable .`          | Daily use. Lightweight (~50 MB) + auto-updating from src. |
| `pip install -e .`                   | Working inside a venv (e.g. for development).             |
| `pip install -e ".[training]"`       | You want to fine-tune your own voices.                    |
| `make binary`                        | Standalone single-file executable for distribution.       |

```bash
make help    # see every target
```

## Workflow

### 1. Inspect the parse (optional, free)

```bash
audiobook inspect /path/to/manuscript.md
```

Prints chapter/scene/word counts. Use this to verify your manuscript is
structured the way the parser expects (`# CHAPTER N`, `# PROLOGUE`,
`---` for scene breaks).

### 2. Review dialogue attribution (multi-voice only)

```bash
audiobook attribute /path/to/manuscript.md
```

Writes `output/attribution.json` with one entry per dialogue line. You
can hand-edit this before the final render if a character is mis-attributed.

### 3. Render a short sample

Always do this first — it surfaces voice / pronunciation issues cheaply.

```bash
audiobook sample /path/to/manuscript.md --mode single --paragraphs 8
audiobook sample /path/to/manuscript.md --mode multi  --paragraphs 8
```

### 4. Render the full book

```bash
audiobook render /path/to/manuscript.md --mode single   # traditional
audiobook render /path/to/manuscript.md --mode multi    # multi-voice
audiobook render /path/to/manuscript.md --mode both     # both passes
```

Output:
```
output/
  single/
    chapters/01_Prologue.mp3, ...
    Book_Title.m4b           # chapter markers, cover, metadata
  multi/...
```

### Or just run it interactively

```bash
audiobook
```

You'll be walked through manuscript selection, mode, backend, output
directory, and cover art via Rich prompts.

## Configuration

Two YAML files in `config/`:

### `config/voices.yaml`

```yaml
narrator:
  voice: bm_george         # Kokoro voice ID
  speed: 0.92
  emotions:                # per-emotion overrides (optional)
    whispered: { speed: 0.78 }
    excited:   { speed: 1.02 }

cast:
  Gael:
    voice: am_puck
    speed: 1.0
    emotions:
      angry:     { speed: 1.05 }
      whispered: { speed: 0.82 }
  # ...
```

Emotions are detected automatically from dialogue tags. Recognized labels:
`neutral, happy, sad, angry, fearful, surprised, disgusted, whispered, excited, calm`.

### `config/pronunciations.yaml`

Whole-word, case-insensitive text substitutions applied before TTS:

```yaml
Gael:      "Gale"
Brenneth:  "Bren-eth"
Dusthollow: "Dust Hollow"
```

## Emotion: how it works

Two layers stacked on top of Kokoro:

1. **Auto-detection from text.** Dialogue tag verbs (whispered, shouted,
   growled, sobbed, gasped, cackled, …) are matched in the narration
   surrounding each dialogue line, plus heuristics for ALL-CAPS shouting
   and `!!`-style excitement.

2. **Per-emotion voice overrides.** For each character you can specify
   different `voice` / `speed` per emotion. Kokoro has no native emotion
   control, so we map emotion to prosody by adjusting these parameters.

For finer-grained, more expressive emotion you train your own
voices. See below.

## Training your own emotion-aware voices (XTTS v2 fine-tuning)

This produces voices that can actually shift their delivery style based
on emotion-conditioning reference clips. Realistic but heavy: you need
labeled audio data and ideally a GPU.

```bash
# Install training extras (adds torch + coqui-tts + librosa; ~3 GB)
pip install -e ".[training]"

# 1. Get a dataset
# Easiest free option: RAVDESS (https://zenodo.org/record/1188976) —
# 24 actors × 8 emotions × 2 sentences. About 1500 clips, 1.4 GB.
# More natural: ESD (https://github.com/HLTSingapore/Emotional-Speech-Data)
# — 10 English speakers × 5 emotions × 350 utterances.

# 2. Ingest into our manifest format
audiobook train ingest --source ravdess --path /path/to/RAVDESS --out data/manifest.csv

# 3. Sanity-check + see stats
audiobook train validate --manifest data/manifest.csv
audiobook train stats    --manifest data/manifest.csv

# 4. Prepare for XTTS (resample, clip, build LJSpeech-style metadata)
audiobook train prepare --manifest data/manifest.csv --out data/prepared

# 5. Fine-tune (hours on GPU, days on MPS, glacial on CPU)
audiobook train run --data data/prepared --out models/my_voices --epochs 10

# 6. Try one line out
audiobook train test --model models/my_voices \
  --speaker RAVDESS_F02 --emotion angry \
  --text "How dare you." --out test.wav

# 7. Use the fine-tuned model in the audiobook pipeline
audiobook render /path/to/manuscript.md --backend xtts --model-dir models/my_voices
```

### What you need to actually train

- **Hardware**: NVIDIA GPU (8 GB+) is ideal. Apple Silicon (MPS) works
  but training will take days for a real fine-tune. CPU is not realistic.
- **Data**: at least ~100 minutes of labeled audio per (speaker, emotion)
  pair for decent quality. RAVDESS gets you started but is read-style,
  not conversational.
- **Time**: ~6–24 hours of training for a useful fine-tune on a GPU.

### Custom datasets

If you have your own recordings, write a manifest yourself:

```csv
audio_path,text,speaker,emotion
recordings/gael_001.wav,"The stones hold.",Gael,calm
recordings/gael_002.wav,"GET BACK!",Gael,angry
recordings/sevet_001.wav,"Listen carefully.",Sevet,whispered
...
```

Then `audiobook train ingest --source custom --path manifest.csv --out data/manifest.csv`
and continue as above.

## Standalone binary

`make binary` produces `dist/audiobook` (~600 MB) using PyInstaller. It
bundles the Kokoro backend but NOT the fine-tuning subsystem (too heavy).
The binary works on the same OS / architecture it was built on.

```bash
make install-binary
make binary
./dist/audiobook               # interactive
./dist/audiobook inspect ...   # subcommand
```

To distribute: copy `dist/audiobook` and `config/` to the target machine.

## Costs and timing

| Step                                     | Cost / time                                   |
|------------------------------------------|-----------------------------------------------|
| Parse + inspect                          | Free, seconds                                 |
| Sample (a few paragraphs)                | Free, ~1–3 min                                |
| Full render (Kokoro, 100k words)         | Free; 4–8 h CPU, <1 h GPU                     |
| Fine-tune XTTS (RAVDESS-sized data)      | Free*; 6–24 h GPU                             |
| Full render (fine-tuned XTTS)            | Free; ~2x slower than Kokoro                  |

\*Apart from electricity. Nothing in this pipeline calls a paid API.

## Project layout

```
audiobook-converter/
├── pyproject.toml, Makefile, audiobook.spec, LICENSE, README.md
├── config/
│   ├── voices.yaml             # voice cast + per-emotion overrides
│   └── pronunciations.yaml     # phonetic spellings for invented names
└── src/audiobook/
    ├── cli.py                  # main `audiobook` CLI
    ├── interactive.py          # guided no-args walkthrough
    ├── parser.py               # markdown -> chapters/scenes/paragraphs
    ├── attribution.py          # rule-based dialogue speaker tagging
    ├── pronounce.py            # phonetic substitution
    ├── emotion.py              # dialogue-tag -> emotion label
    ├── synth.py                # backend factory + voice cast loader
    ├── stitch.py               # assemble paragraphs into chapter audio
    ├── package.py              # M4B with chapter markers + cover art
    ├── backends/
    │   ├── base.py             # Backend protocol
    │   ├── kokoro.py           # Kokoro TTS implementation
    │   └── xtts.py             # fine-tuned XTTS adapter
    └── training/
        ├── dataset.py          # RAVDESS / ESD / custom ingest + prepare
        ├── train.py            # XTTS v2 fine-tune runner
        ├── infer.py            # FineTunedXTTSSynth
        └── cli.py              # `audiobook train ...` subcommands
```

## Troubleshooting

- **`ffmpeg not found`** — `brew install ffmpeg`.
- **`espeak-ng: command not found`** — `brew install espeak-ng`.
- **`No module named 'kokoro'`** — install: `pipx install --editable .`
  or activate your venv first.
- **Multi-voice attributing a major character to `_OTHER`** — that's
  the fallback for alternation partners whose name was never given in
  narration. Edit `output/attribution.json` to fix, or add the missing
  character to `cast:` in `voices.yaml`.
- **`audiobook train run` says `coqui-tts is required`** — install
  training extras: `pip install -e ".[training]"`.
- **PyInstaller binary crashes on first run** — usually a missing data
  file. Re-build with `make clean && make binary`. If still broken,
  check that `espeak-ng` is on the runtime PATH on the target machine.

## License

MIT. See [LICENSE](LICENSE).
