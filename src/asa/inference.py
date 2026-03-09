"""
inference.py — Load a trained Qwen2-Audio checkpoint and run inference.

Public API
----------
load_model   : load processor + model from a Hub repo or local directory
run_inference : generate text responses for a list of audio files (supports A/B mode)
"""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import typer
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from asa.data import PROMPT_TEMPLATE, PROMPT_TEMPLATE_AB, load_audio


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class ASAModel(str):
    """Hub model IDs for ASA checkpoints.

    Being a plain ``str`` subclass means you can pass these directly to
    :func:`load_model` (or any ``from_pretrained`` call) without ``.value``.

    Usage::

        processor, model, device = load_model(ASAModel.SFT)
        processor, model, device = load_model(ASAModel.AB)
    """

    # Full supervised fine-tune (standard single-audio quality assessment)
    SFT = "Leng2beat/speech-quality-assessement-qwen2audio-full-sft"

    # A/B preference model (two-audio comparative quality assessment)
    SFT_AB = "Leng2beat/speech-quality-assessement-qwen2audio-sft-ab"  # TODO: update when pushed
    # AALD distillation variants
    AALD_AB = "Leng2beat/speech-quality-assessement-qwen2audio-aald-ab"  # TODO: update when pushed
    AALD = "Leng2beat/speech-quality-assessement-qwen2audio-aald"  # TODO: update when pushed


def load_model(
    model_id: str | Path,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoProcessor, Qwen2AudioForConditionalGeneration, str]:
    """Load processor + model from a Hugging Face Hub repo or local directory.

    Parameters
    ----------
    model_id  : Hub repo ID (e.g. ``"your-org/qwen2-audio-sft"``) **or**
                path to a local checkpoint directory (e.g. ``results/sft``).
    device    : ``"cuda"`` / ``"cpu"``.  Defaults to CUDA when available.
    dtype     : override the stored weight dtype (e.g. ``torch.float32``).

    Returns
    -------
    processor, model, device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
        low_cpu_mem_usage=True,
    )
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)
    return processor, model, device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    model: Qwen2AudioForConditionalGeneration,
    processor: AutoProcessor,
    audio_paths: Iterable[str | Path] | Iterable[Tuple[str | Path, str | Path]],
    ab_mode: bool = False,
    device: Optional[str] = None,
    max_new_tokens: int = 100,
    batch_size: int = 4,
) -> List[str]:
    """Generate text responses for a list of audio files.

    Supports both single-audio and A/B (two-audio) inference modes.
    Prompts are loaded automatically from :mod:`asa.data`.

    Parameters
    ----------
    audio_paths    : In normal mode — paths to ``.wav`` files.
                     In A/B mode — an iterable of ``(path_a, path_b)`` tuples.
    ab_mode        : When ``True``, treat ``audio_paths`` as pairs
                     (A/B comparative preference format).
    device         : Inferred from model parameters if not given.
    max_new_tokens : Generation budget per sample.
    batch_size     : Number of files (or pairs) to process at once.

    Returns
    -------
    List of decoded model responses (prompt tokens are stripped).
    """
    if device is None:
        device = next(model.parameters()).device.type

    audio_paths = list(audio_paths)
    sr = processor.feature_extractor.sampling_rate
    all_responses: List[str] = []

    if ab_mode:
        # ------------------------------------------------------------------
        # A/B mode: each element of audio_paths is a (path_a, path_b) tuple
        # ------------------------------------------------------------------
        prompts = [PROMPT_TEMPLATE_AB] * len(audio_paths)

        for start in tqdm(
            range(0, len(audio_paths), batch_size), desc="Running A/B inference"
        ):
            batch_pairs = audio_paths[start : start + batch_size]
            batch_prompts = prompts[start : start + batch_size]

            audios = []
            for audio_a, audio_b in batch_pairs:
                audios.append(load_audio(str(audio_a), target_sr=sr))
                audios.append(load_audio(str(audio_b), target_sr=sr))

            batch = processor(
                text=batch_prompts,
                audio=audios,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            batch = {k: v.to(device) for k, v in batch.items()}

            input_len = batch["input_ids"].shape[1]

            with torch.no_grad():
                out_ids = model.generate(**batch, max_new_tokens=max_new_tokens)

            response_ids = out_ids[:, input_len:]
            decoded = processor.batch_decode(response_ids, skip_special_tokens=True)
            all_responses.extend(decoded)

    else:
        # ------------------------------------------------------------------
        # Normal mode: each element of audio_paths is a single file path
        # ------------------------------------------------------------------
        texts = [PROMPT_TEMPLATE] * len(audio_paths)

        for start in tqdm(
            range(0, len(audio_paths), batch_size), desc="Running inference"
        ):
            batch_paths = audio_paths[start : start + batch_size]
            batch_texts = texts[start : start + batch_size]

            audios = [load_audio(str(p), target_sr=sr) for p in batch_paths]

            batch = processor(
                text=batch_texts,
                audio=audios,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            batch = {k: v.to(device) for k, v in batch.items()}

            input_len = batch["input_ids"].shape[1]

            with torch.no_grad():
                out_ids = model.generate(**batch, max_new_tokens=max_new_tokens)

            # Strip prompt tokens so we only return the model's response
            response_ids = out_ids[:, input_len:]
            decoded = processor.batch_decode(response_ids, skip_special_tokens=True)
            all_responses.extend(decoded)

    return all_responses


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(help="Run ASA inference on audio files.")


@app.command()
def infer(
    audio_paths: List[Path] = typer.Argument(
        ..., help="Paths to .wav files to evaluate."
    ),
    model_id: str = typer.Option(
        ASAModel.SFT, "--model", "-m", help="Hub repo ID or local checkpoint path."
    ),
    ab_mode: bool = typer.Option(
        False, "--ab", help="A/B mode: audio_paths must be pairs (a1 b1 a2 b2 ...)."
    ),
    max_new_tokens: int = typer.Option(100, help="Max tokens to generate per sample."),
    batch_size: int = typer.Option(4, help="Batch size."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write responses to a text file (one per line)."
    ),
):
    """Load a model and run inference on the given audio files."""
    processor, model, device = load_model(model_id)

    if ab_mode:
        if len(audio_paths) % 2 != 0:
            typer.echo(
                "ERROR: --ab requires an even number of audio paths (pairs).", err=True
            )
            raise typer.Exit(1)
        pairs = list(zip(audio_paths[::2], audio_paths[1::2]))
        responses = run_inference(
            model,
            processor,
            pairs,
            ab_mode=True,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
    else:
        responses = run_inference(
            model,
            processor,
            audio_paths,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )

    for path, response in zip(
        audio_paths if not ab_mode else [f"{a} / {b}" for a, b in pairs], responses
    ):
        typer.echo(f"{path}: {response}")

    if output:
        output.write_text("\n".join(responses))
        typer.echo(f"\nResponses written to {output}")


if __name__ == "__main__":
    app()
