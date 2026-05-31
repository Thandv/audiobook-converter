# PyInstaller spec for the `audiobook` standalone binary.
#
# Build with:  make binary
# Output:      dist/audiobook (or dist/audiobook.exe on Windows)
#
# Note: this bundles the audiobook code, config, and Python deps — but
# NOT the Kokoro model weights (those download on first run to ~/.cache).
# Bundled binary is ~600 MB; with model weights bundled it would be ~1 GB.
# See `binaries_to_include` below to switch behavior.

# ruff: noqa
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules
from pathlib import Path

block_cipher = None

# Kokoro and its sub-dependencies need their submodules and data files
# pulled in explicitly because PyInstaller can't always find them through
# imports alone.
hidden = []
datas = []
binaries = []

for pkg in ("kokoro", "phonemizer", "espeakng_loader", "loguru", "rich", "click"):
    try:
        b, d, h = collect_all(pkg)
        binaries += b
        datas += d
        hidden += h
    except Exception as e:
        print(f"[spec] skipping missing package {pkg}: {e}")

# Ship our YAML config files alongside the binary.
config_root = Path("config")
if config_root.exists():
    for f in config_root.iterdir():
        if f.is_file():
            datas.append((str(f), "config"))

a = Analysis(
    ["src/audiobook/__main__.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden + collect_submodules("audiobook"),
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Skip the training stack — too heavy. The binary doesn't ship
        # the fine-tuning subsystem; install via pipx for that.
        "torch",
        "torchaudio",
        "TTS",
        "trainer",
        "librosa",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="audiobook",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
