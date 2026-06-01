"""
inference.py — Load a trained Qwen2-Audio checkpoint and run inference.

Public API
----------
load_model   : load processor + model from a Hub repo or local directory
run_inference : generate text responses for a list of audio files
"""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import typer
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
)

from asa.audio import load_audio
from asa.modeling_timeaudio import Qwen2AudioTimeForConditionalGeneration
from asa.prompts import PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class ASAModel(str):
    """Hub model IDs for ASA checkpoints.

    Being a plain ``str`` subclass means you can pass these directly to
    :func:`load_model` (or any ``from_pretrained`` call) without ``.value``.

    Usage::

        processor, model, device = load_model(ASAModel.SFT)
    """

    # Full supervised fine-tune (standard single-audio quality assessment)
    SFT = "Leng2beat/speech-quality-assessement-qwen2audio-full-sft"

    # AALD distillation variant
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

    # TimeAudio checkpoints must load through the subclass. Two independent
    # markers force this: use_abs_time_embedding (extra abs_time_embedding param)
    # and use_time_tokens (extended vocab + resized lm_head). Either alone makes
    # the stock class fail or silently mis-load, so detect both. Stock
    # checkpoints have neither flag and are unaffected.
    try:
        cfg = AutoConfig.from_pretrained(model_id)
        is_timeaudio = bool(
            getattr(cfg, "use_abs_time_embedding", False)
            or getattr(cfg, "use_time_tokens", False)
        )
    except Exception:
        is_timeaudio = False

    model_cls = (
        Qwen2AudioTimeForConditionalGeneration
        if is_timeaudio
        else Qwen2AudioForConditionalGeneration
    )
    model = model_cls.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device.startswith("cuda") else None,
        low_cpu_mem_usage=True,
    )
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)

    # Force left-padding for generation. Qwen2-Audio is decoder-only, so batched
    # generation must left-pad: right-padding makes generation start after the
    # pad tokens and the model emits EOS immediately, yielding empty output.
    # DPO checkpoints save the tokenizer with padding_side="right" (the training
    # default), so without this override every batched eval of a DPO checkpoint
    # returns empty strings. SFT checkpoints happen to store "left" and are
    # unaffected; setting it unconditionally is correct for all generation.
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"

    return processor, model, device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    model: Qwen2AudioForConditionalGeneration,
    processor: AutoProcessor,
    audio_paths: Iterable[str | Path],
    prompt_texts: Optional[List[str]] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 100,
    batch_size: int = 4,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    skip_special_tokens: bool = True,
) -> List[str]:
    """Generate text responses for a list of audio files.

    Single-audio inference. Prompts are loaded automatically from :mod:`asa.data`
    when ``prompt_texts`` is not supplied.

    Parameters
    ----------
    audio_paths    : Paths to ``.wav`` files.
    prompt_texts   : Optional per-sample prompts. When provided, must match
                     ``len(audio_paths)``. Falls back to the default prompt
                     template when omitted.
    device         : Inferred from model parameters if not given.
    max_new_tokens : Generation budget per sample.
    batch_size     : Number of files (or pairs) to process at once.
    do_sample      : When ``True``, enable stochastic decoding.
    temperature    : Softmax temperature used when ``do_sample`` is ``True``.
    top_p          : Nucleus sampling cutoff used when ``do_sample`` is ``True``.
    skip_special_tokens
                   : Passed to processor decoding. Set ``False`` for tasks that
                     need generated timestamp tokens such as ``<|2.88|>``.

    Returns
    -------
    List of decoded model responses (prompt tokens are stripped).
    """
    if device is None:
        device = next(model.parameters()).device.type

    audio_paths = list(audio_paths)
    sr = processor.feature_extractor.sampling_rate
    all_responses: List[str] = []
    if prompt_texts is not None and len(prompt_texts) != len(audio_paths):
        raise ValueError(
            "prompt_texts length must match audio_paths length. "
            f"Got prompt_texts={len(prompt_texts)} and audio_paths={len(audio_paths)}."
        )

    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if do_sample:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )

    texts = (
        prompt_texts
        if prompt_texts is not None
        else [PROMPT_TEMPLATE] * len(audio_paths)
    )

    for start in tqdm(
        range(0, len(audio_paths), batch_size), desc="Running inference"
    ):
        batch_paths = audio_paths[start : start + batch_size]
        batch_texts = texts[start : start + batch_size]

        audios = [load_audio(str(p), target_sr=sr) for p in batch_paths]

        batch = processor(
            text=batch_texts,
            audios=audios,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        input_len = batch["input_ids"].shape[1]

        with torch.no_grad():
            out_ids = model.generate(**batch, **gen_kwargs)

        # Strip prompt tokens so we only return the model's response
        response_ids = out_ids[:, input_len:]
        decoded = processor.batch_decode(
            response_ids, skip_special_tokens=skip_special_tokens
        )
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
    max_new_tokens: int = typer.Option(150, help="Max tokens to generate per sample."),
    batch_size: int = typer.Option(4, help="Batch size."),
    do_sample: bool = typer.Option(
        True,
        "--do-sample/--greedy",
        help="Sample with temperature/top_p (default) or greedy decoding.",
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", help="Sampling temperature (used when --do-sample)."
    ),
    top_p: float = typer.Option(
        0.9, "--top-p", help="Nucleus top-p cutoff (used when --do-sample)."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write responses to a text file (one per line)."
    ),
):
    """Load a model and run inference on the given audio files."""
    processor, model, device = load_model(model_id)

    responses = run_inference(
        model,
        processor,
        audio_paths,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    for path, response in zip(audio_paths, responses):
        typer.echo(f"{path}: {response}")

    if output:
        output.write_text("\n".join(responses))
        typer.echo(f"\nResponses written to {output}")


if __name__ == "__main__":
    app()
