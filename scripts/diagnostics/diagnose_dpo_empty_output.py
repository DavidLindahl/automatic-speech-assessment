"""Diagnostic: load the DPO model and try generating from one test sample.

Goal: figure out WHY all eval predictions are empty strings while the model
loads, runs without errors, and the eval script is the same one that works
for the SFT-warmup checkpoint.

Hypotheses to test (in order):
  A. Model emits EOS at step 0 -> output is empty before slicing.
  B. Slicing logic in eval (out_ids[:, input_len:]) over-strips because
     generation re-emitted the prompt prefix.
  C. Something else - we'll see the raw output and decide.

Usage:
    uv run python scripts/diagnostics/diagnose_dpo_empty_output.py \
        --model models/dpo_paper_half_h100 \
        --dataset data/processed/eval/test_LIVE.json \
        --num 3
"""

from __future__ import annotations

from pathlib import Path

import torch
import typer

from asa.data import PROMPT_TEMPLATE, TARGET_SR
from asa.inference import load_audio
from asa.processed_data import load_processed_records, resolve_audio_path
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


app = typer.Typer()


@app.command()
def diagnose(
    model: str = typer.Option(
        "models/dpo_paper_half_h100", help="Path or hub id of model under test."
    ),
    dataset: str = typer.Option(
        "data/processed/eval/test_LIVE.json", help="Test dataset JSON."
    ),
    data_root: Path = typer.Option(Path("data"), help="Root for audio paths."),
    num: int = typer.Option(3, help="How many samples to probe."),
    max_new_tokens: int = typer.Option(50, help="Generation cap."),
):
    print(f"=== Loading {model} ===")
    processor = AutoProcessor.from_pretrained(model)
    tok = processor.tokenizer
    # Match the fixed eval: decoder-only generation needs left-padding.
    tok.padding_side = "left"
    m, loading_info = Qwen2AudioForConditionalGeneration.from_pretrained(
        model, torch_dtype=torch.bfloat16, output_loading_info=True
    )
    m = m.to("cuda").eval()

    print(f"  missing_keys        = {loading_info.get('missing_keys')}")
    print(f"  unexpected_keys     = {loading_info.get('unexpected_keys')}")
    print(f"  mismatched_keys     = {loading_info.get('mismatched_keys')}")
    print(f"  padding_side        = {tok.padding_side}")
    print(f"  pad_token_id        = {tok.pad_token_id}  ({tok.pad_token!r})")
    print(f"  eos_token_id        = {tok.eos_token_id}  ({tok.eos_token!r})")
    print(f"  bos_token_id        = {tok.bos_token_id}")
    print(f"  generation_config.eos_token_id  = {m.generation_config.eos_token_id}")
    print(f"  generation_config.pad_token_id  = {m.generation_config.pad_token_id}")
    print(f"  PROMPT_TEMPLATE     = {PROMPT_TEMPLATE!r}")

    records = load_processed_records(Path(dataset))[:num]
    print(f"\n=== Probing {len(records)} samples ===")

    for i, rec in enumerate(records):
        print(f"\n--- sample {i}: true MOS = {rec.get('mos')} ---")
        audio_path = resolve_audio_path(rec["audios"][0], data_root)
        audio = load_audio(str(audio_path), target_sr=TARGET_SR)

        batch = processor(
            text=[PROMPT_TEMPLATE],
            audios=[audio],
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        batch = {k: v.to("cuda") for k, v in batch.items()}
        input_len = batch["input_ids"].shape[1]
        print(
            f"  input_ids shape: {tuple(batch['input_ids'].shape)}  (len={input_len})"
        )
        print(f"  prompt last 8 ids: {batch['input_ids'][0, -8:].tolist()}")
        print(f"  prompt decoded:    {tok.decode(batch['input_ids'][0])[-200:]!r}")

        with torch.no_grad():
            out = m.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        seq = out.sequences
        print(f"  output_ids shape:  {tuple(seq.shape)}")
        print(f"  generated len:     {seq.shape[1] - input_len}")
        print(f"  full output last 16 ids: {seq[0, -16:].tolist()}")

        # Without slicing
        decoded_full = tok.decode(seq[0])
        # With the eval script's slicing
        new_ids = seq[0, input_len:]
        decoded_new = tok.decode(new_ids)
        decoded_new_skip = tok.decode(new_ids, skip_special_tokens=True)

        print(f"  decoded full   (last 200): {decoded_full[-200:]!r}")
        print(f"  decoded sliced (raw):      {decoded_new!r}")
        print(f"  decoded sliced (skip_sp):  {decoded_new_skip!r}")

        # First-step probability mass on EOS
        first_logits = out.scores[0][0]
        probs = torch.softmax(first_logits.float(), dim=-1)
        eos_ids = m.generation_config.eos_token_id
        if not isinstance(eos_ids, list):
            eos_ids = [eos_ids]
        for eid in eos_ids:
            print(f"  P(token={eid}) at step 0: {probs[eid].item():.4f}")
        topk = torch.topk(probs, 5)
        print(
            "  top-5 step-0 ids/probs: "
            + ", ".join(
                f"{int(i)}={p:.3f}({tok.decode([int(i)])!r})"
                for i, p in zip(topk.indices.tolist(), topk.values.tolist())
            )
        )


if __name__ == "__main__":
    app()
