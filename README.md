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

Three escalating layers — pick the one whose setup cost matches the
emotional fidelity you want.

### Layer 1 — Kokoro + per-emotion voice/speed overrides (free, no setup)

- Dialogue-tag verbs (whispered, shouted, growled, sobbed, gasped, …)
  are auto-detected in the narration around each dialogue line, plus
  heuristics for ALL-CAPS shouting and `!!`-style excitement.
- Each character can map each emotion to a different Kokoro voice ID
  and/or speed in `config/voices.yaml`. The system already ships with
  sensible per-emotion speed adjustments.

Limit: Kokoro itself has no emotion knob — this is prosody nudging, not
emotional acting. Sounds *competent*, not *moving*.

### Layer 2 — Voice library + zero-shot cloning (free; ~10 min setup)

This is where voices actually get emotional. You give the system a
short audio clip per (character, emotion), e.g. a 10-second
`voices/Gael/angry.wav`. At render time, XTTS v2 or Chatterbox **clones
the voice AND copies the emotional prosody** from that clip into the
new line. No training, no GPU strictly required.

The fastest way to get going is the RAVDESS bulk-import — it gives you
emotional reference clips for every emotion in under 5 minutes:

```bash
# 0. Install the cloning extras (~3 GB; one-time)
pip install -e ".[training]"          # for XTTS cloning backend
pip install -e ".[chatterbox]"        # for Chatterbox backend

# 1. Download RAVDESS Speech (free, ~1.4 GB)
#    https://zenodo.org/record/1188976
unzip Audio_Speech_Actors_01-24.zip -d ~/datasets/RAVDESS

# 2. Bulk-import as the library's stock emotion voices
audiobook voices import-ravdess --path ~/datasets/RAVDESS

# 3. (Optional) Map specific RAVDESS actors to your characters
audiobook voices import-ravdess --path ~/datasets/RAVDESS \
  --map "Gael=01,Sera=02,Brenneth=04,narrator=11"

# 4. Render with the cloning backend
audiobook render /path/to/manuscript.md \
  --backend cloning --library voices/

# Or with Chatterbox (explicit emotion intensity knob)
audiobook render /path/to/manuscript.md \
  --backend chatterbox --library voices/
```

To replace any stock clip with your own voice:

```bash
# Record live from your mic (templates printed on-screen to read aloud)
audiobook voices record --character Gael --emotion angry

# Or import an existing audio file
audiobook voices import \
  --character Gael --emotion sad \
  --file ~/recordings/gael_sad.wav --overwrite

# See what's in the library
audiobook voices list
audiobook voices show Gael
audiobook voices coverage           # what's missing
audiobook voices validate           # any clips below recommended length / sample rate?
audiobook voices templates          # sample sentences to read for each emotion
```

**XTTS cloning vs Chatterbox** — both are good; pick by what you have:

| | XTTS cloning | Chatterbox |
|---|---|---|
| Best with | Emotion-specific clips per character | One neutral clip per character |
| Emotion intensity | Copied from reference prosody | Explicit 0-2 knob per emotion |
| Quality | Very good | Very good, more theatrical |
| Setup size | ~3 GB | ~2 GB |
| Render speed | Moderate | Faster |

### Layer 3 — Fine-tune your own XTTS model (free; GPU + hours)

For the absolute best quality and full control, fine-tune XTTS v2 on
your own emotion-labeled data. See *Training your own emotion-aware
voices* below — this is overkill for most projects.

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
    ├── voice_library.py        # voices/<character>/<emotion>.wav tree
    ├── voice_cli.py            # `audiobook voices ...` subcommands
    ├── synth.py                # backend factory + voice cast loader
    ├── stitch.py               # assemble paragraphs into chapter audio
    ├── package.py              # M4B with chapter markers + cover art
    ├── backends/
    │   ├── base.py             # Backend protocol
    │   ├── kokoro.py           # Kokoro TTS implementation
    │   ├── xtts.py             # fine-tuned XTTS adapter
    │   ├── cloning.py          # zero-shot XTTS v2 voice cloning
    │   └── chatterbox.py       # Chatterbox with emotion intensity
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
