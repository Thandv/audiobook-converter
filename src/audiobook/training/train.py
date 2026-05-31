"""Fine-tune Coqui XTTS v2 on a prepared dataset.

This module wraps the community-maintained Coqui fork
(``github.com/idiap/coqui-ai-TTS``, PyPI: ``coqui-tts``). Heavy imports are
deferred to ``fine_tune_xtts`` so importing the audiobook package without
training extras stays cheap.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from loguru import logger


XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_LANGUAGE = "en"


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def _autodetect_device() -> str:
    """Pick the best available device: CUDA > MPS > CPU."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for training. Install training extras: "
            'pip install -e ".[training]"'
        ) from exc

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_device(device: str) -> str:
    if device == "auto":
        chosen = _autodetect_device()
    else:
        chosen = device
    if chosen == "cpu":
        logger.warning(
            "Training on CPU will be glacial. Consider a CUDA GPU or "
            "Apple Silicon (MPS) device."
        )
    else:
        logger.info("Training device: {}", chosen)
    return chosen


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(log_path: Path) -> None:
    """Tee loguru output to ``log_path`` while keeping stdout live."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # loguru's default sink is stderr; add our file sink without removing stdout.
    logger.add(
        str(log_path),
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    # Mirror to stdout too for live progress.
    logger.add(sys.stdout, level="INFO", colorize=True, filter=_dedupe_stdout)


_seen_stdout_ids: set[int] = set()


def _dedupe_stdout(record: dict[str, Any]) -> bool:
    """Avoid duplicating records that loguru already sent to stderr default."""
    # Each record has an id-like sink_id we can't access; use the raw object id.
    rid = id(record)
    if rid in _seen_stdout_ids:
        return False
    _seen_stdout_ids.add(rid)
    return True


# ---------------------------------------------------------------------------
# Reference clip selection
# ---------------------------------------------------------------------------


def _copy_reference_clips(prepared_data_dir: Path, output_dir: Path) -> Path:
    """Copy ``prepared_data_dir/reference_clips`` into ``output_dir``.

    The inference path looks for these clips alongside the checkpoint.
    """
    src = prepared_data_dir / "reference_clips"
    dst = output_dir / "reference_clips"
    if not src.exists():
        logger.warning("No reference_clips/ in prepared data; inference will need manual refs.")
        return dst
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    n = sum(1 for _ in dst.rglob("*.wav"))
    logger.info("Copied {} reference clip(s) to {}", n, dst)
    return dst


# ---------------------------------------------------------------------------
# Fine-tune entry point
# ---------------------------------------------------------------------------


def fine_tune_xtts(
    prepared_data_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 5e-6,
    device: str = "auto",
) -> Path:
    """Fine-tune XTTS v2 on prepared training data.

    Parameters
    ----------
    prepared_data_dir:
        Output of :func:`audiobook.training.dataset.prepare_for_xtts`.
    output_dir:
        Destination for checkpoints, config, reference clips, and logs.
    epochs, batch_size, lr:
        Training hyperparameters. Defaults are conservative for fine-tuning.
    device:
        ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Returns the path to ``best_model.pth``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir / "training.log")
    logger.info("Starting XTTS v2 fine-tune: data={} out={}", prepared_data_dir, output_dir)

    resolved_device = _resolve_device(device)

    # --- Lazy imports: Coqui stack is huge; defer until we actually train. --
    try:
        import torch  # type: ignore[import-not-found]  # noqa: F401
        from trainer import Trainer, TrainerArgs  # type: ignore[import-not-found]
        from TTS.config.shared_configs import BaseDatasetConfig  # type: ignore[import-not-found]
        from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore[import-not-found]
        from TTS.tts.datasets import load_tts_samples  # type: ignore[import-not-found]
        from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig  # type: ignore[import-not-found]
        from TTS.utils.manage import ModelManager  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "coqui-tts is required for fine-tuning. Install training extras: "
            'pip install -e ".[training]"'
        ) from exc

    train_meta = prepared_data_dir / "metadata_train.csv"
    eval_meta = prepared_data_dir / "metadata_eval.csv"
    wavs_dir = prepared_data_dir / "wavs"
    if not train_meta.exists() or not wavs_dir.exists():
        raise FileNotFoundError(
            f"Prepared data missing metadata_train.csv or wavs/ in {prepared_data_dir}"
        )

    # Download base model files (idempotent — ModelManager caches).
    manager = ModelManager()
    model_path, config_path, _model_item = manager.download_model(XTTS_MODEL_NAME)
    base_model_dir = Path(model_path)
    base_config_path = Path(config_path)
    logger.info("Base XTTS model dir: {}", base_model_dir)

    # Load + override base config for fine-tuning.
    config = XttsConfig()
    config.load_json(str(base_config_path))
    config.audio = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )
    config.model_args = XttsArgs(
        max_conditioning_length=132300,  # 6s @ 22.05kHz
        min_conditioning_length=66150,   # 3s
        max_wav_length=255995,           # ~11.6s
        max_text_length=200,
        gpt_max_audio_tokens=605,
        gpt_max_text_tokens=402,
    )
    config.batch_size = batch_size
    config.eval_batch_size = max(1, batch_size // 2)
    config.num_loader_workers = 2
    config.num_eval_loader_workers = 1
    config.epochs = epochs
    config.lr = lr
    config.run_eval = eval_meta.exists() and eval_meta.stat().st_size > 0
    config.output_path = str(output_dir)
    config.language = DEFAULT_LANGUAGE
    config.save_step = 1000
    config.save_n_checkpoints = 2
    config.save_checkpoints = True
    config.print_step = 50
    config.print_eval = True
    config.mixed_precision = resolved_device == "cuda"

    # Dataset descriptor (LJSpeech-style with our metadata files).
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="audiobook_emotion_set",
        path=str(prepared_data_dir),
        meta_file_train=str(train_meta),
        meta_file_val=str(eval_meta) if config.run_eval else "",
        language=DEFAULT_LANGUAGE,
    )
    config.datasets = [dataset_config]

    # Persist the active config for inference.
    config.save_json(str(output_dir / "config.json"))

    # Build & initialise the model from the base checkpoint.
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=str(base_model_dir),
        eval=False,
        strict=False,
    )

    # Load training samples.
    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=config.run_eval,
    )
    logger.info(
        "Loaded {} train samples / {} eval samples",
        len(train_samples),
        len(eval_samples) if eval_samples else 0,
    )

    # Resume from last checkpoint if one exists in output_dir.
    restore_path = _find_resume_checkpoint(output_dir)
    if restore_path is not None:
        logger.info("Resuming from checkpoint: {}", restore_path)

    trainer_args = TrainerArgs(
        restore_path=str(restore_path) if restore_path else None,
        skip_train_epoch=False,
        start_with_eval=False,
        grad_accum_steps=1,
    )

    trainer = Trainer(
        trainer_args,
        config,
        output_path=str(output_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    best_ckpt = _locate_best_checkpoint(output_dir)
    final_path = output_dir / "best_model.pth"
    if best_ckpt is not None and best_ckpt != final_path:
        shutil.copy2(best_ckpt, final_path)
    if not final_path.exists():
        raise RuntimeError(
            f"Training finished but no best_model.pth found under {output_dir}"
        )

    # Ship reference clips next to the checkpoint for inference.
    _copy_reference_clips(prepared_data_dir, output_dir)

    # Write a small manifest of what's in the model dir, for the inference loader.
    _write_inference_manifest(output_dir)

    logger.info("Fine-tune complete. Best checkpoint: {}", final_path)
    return final_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_resume_checkpoint(output_dir: Path) -> Path | None:
    """Trainer creates ``output_dir/run-<timestamp>/checkpoint_<step>.pth`` files.

    Pick the highest-step checkpoint in the most recent run, if any.
    """
    runs = sorted(
        [p for p in output_dir.glob("run-*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        ckpts = sorted(run.glob("checkpoint_*.pth"))
        if ckpts:
            return ckpts[-1]
    return None


def _locate_best_checkpoint(output_dir: Path) -> Path | None:
    """Find ``best_model.pth`` (or fall back to the last checkpoint)."""
    direct = output_dir / "best_model.pth"
    if direct.exists():
        return direct
    runs = sorted(
        [p for p in output_dir.glob("run-*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        best = run / "best_model.pth"
        if best.exists():
            return best
        ckpts = sorted(run.glob("checkpoint_*.pth"))
        if ckpts:
            return ckpts[-1]
    return None


def _write_inference_manifest(output_dir: Path) -> None:
    """Write ``inference.json`` describing what's available in the model dir."""
    refs = output_dir / "reference_clips"
    speakers: dict[str, list[str]] = {}
    if refs.exists():
        for sp_dir in sorted(p for p in refs.iterdir() if p.is_dir()):
            emos = sorted(p.name for p in sp_dir.iterdir() if p.is_dir())
            speakers[sp_dir.name] = emos
    payload = {
        "checkpoint": "best_model.pth",
        "config": "config.json",
        "speakers": speakers,
        "native_sample_rate": 24000,
    }
    (output_dir / "inference.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
