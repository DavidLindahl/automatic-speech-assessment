"""Diagnostic: inspect the actual labels tensor produced by ALLDDPOCollator.

The 28374130 diagnostic showed the trained DPO policy puts ~0.5 mass on EOS
at the first generation step. Hypothesis: the collator's _build_labels logic
is misaligned under left-padding - it computes prompt_len from a separately-
padded prompt-only batch, then masks labels[:prompt_len] in the full batch.
With left-padding, the actual prompt occupies positions [pad_count : seq_len-resp_len]
not [0 : prompt_len], so the mask hits the wrong tokens.

This script:
  1. Loads the SFT-warmup checkpoint as the policy model (same starting point
     as the failed DPO run used).
  2. Builds a real ALLDDPOCollator using the cleaned DPO dataset.
  3. Runs the collator on a tiny batch (B=2 -> 4 rows after chosen/rejected
     concatenation).
  4. Prints, for each row:
       - tokenizer.padding_side
       - sequence length
       - actual prompt span (where <|audio_eos|>... ends)
       - prompt_len computed by the collator
       - the slice of labels that's NOT masked (-100), with decoded text
  5. Flags rows where the unmasked label region overlaps with prompt tokens
     (which is the bug).

Runs CPU-only - no GPU needed, no model weights loaded for the policy
forward pass.

    uv run python scripts/diagnostics/diagnose_dpo_label_mask.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import typer
from transformers import AutoProcessor, AutoTokenizer

from asa.data import ALLDDPOCollator, DPODataset


app = typer.Typer()


@app.command()
def diagnose(
    sft_model: str = typer.Option(
        "models/sft_warmup_paper_half_h100",
        help="SFT checkpoint - we only need its processor.",
    ),
    ref_model: str = typer.Option(
        "Qwen/Qwen2-7B", help="Reference model id - we only need its tokenizer."
    ),
    json_path: Path = typer.Option(
        Path("data/processed/train_dpo_paper_half_h100_clean.json"),
        help="DPO training data.",
    ),
    data_root: Path = typer.Option(Path("data")),
    n: int = typer.Option(2, help="Batch size to probe (will produce 2N rows)."),
):
    print(f"Loading audio processor from {sft_model}")
    audio_processor = AutoProcessor.from_pretrained(sft_model)
    tok = audio_processor.tokenizer
    print(f"  tokenizer.padding_side = {tok.padding_side!r}")
    print(f"  tokenizer.pad_token    = {tok.pad_token!r} (id={tok.pad_token_id})")
    print(f"  tokenizer.eos_token    = {tok.eos_token!r} (id={tok.eos_token_id})")

    print(f"\nLoading text tokenizer from {ref_model}")
    text_tokenizer = AutoTokenizer.from_pretrained(ref_model)

    print(f"\nLoading DPO dataset from {json_path}")
    dataset = DPODataset(json_path=json_path, data_root=data_root, max_samples=n)

    collator = ALLDDPOCollator(
        audio_processor=audio_processor, text_tokenizer=text_tokenizer
    )

    features = [dataset[i] for i in range(n)]
    print(f"\nCollating batch of size {n} (will produce 2N={2 * n} rows)")
    batch = collator(features)

    input_ids = batch["policy_input_ids"]
    labels = batch["policy_labels"]
    attn = batch["policy_attention_mask"]

    print(f"\npolicy_input_ids.shape = {tuple(input_ids.shape)}")
    print(f"policy_labels.shape    = {tuple(labels.shape)}")

    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id
    audio_eos_id = tok.convert_tokens_to_ids("<|audio_eos|>")

    print(f"\naudio_eos_token_id = {audio_eos_id}")
    print()

    found_bug = False
    for i in range(input_ids.shape[0]):
        ids = input_ids[i]
        lab = labels[i]
        am = attn[i]

        seq_len = ids.shape[0]
        n_pad = (ids == pad_id).sum().item()
        n_real = (ids != pad_id).sum().item()

        # Where is the audio_eos token? That marks the end of the prompt
        # (training prompt is "<|audio_bos|><|AUDIO|>...<|audio_eos|>Please describe...").
        # Note: prompt also contains "Please describe and evaluate the synthetic speech."
        # after audio_eos, then chosen/rejected text begins.
        audio_eos_positions = (ids == audio_eos_id).nonzero(as_tuple=True)[0].tolist()

        # Where are non-masked labels (i.e. where the loss is computed)?
        unmasked = (lab != -100).nonzero(as_tuple=True)[0].tolist()
        unmasked_start = unmasked[0] if unmasked else None
        unmasked_end = unmasked[-1] if unmasked else None

        # The "true" prompt end: position of audio_eos + length of "Please describe...".
        # We can find this empirically: the chosen/rejected text follows immediately.
        # Decode the unmasked region for inspection.
        first_unmasked_id = ids[unmasked_start].item() if unmasked_start is not None else None
        unmasked_decoded = tok.decode(ids[unmasked_start : unmasked_end + 1]) if unmasked else ""

        # Decode the WHOLE row to anchor visually, with markers for masking.
        row_decoded_left = tok.decode(ids[: max(0, unmasked_start)]) if unmasked_start else tok.decode(ids)

        print(f"--- row {i} ---")
        print(f"  seq_len={seq_len}  n_pad={n_pad}  n_real={n_real}")
        print(f"  attn first 4: {am[:4].tolist()}  attn last 4: {am[-4:].tolist()}")
        print(f"  audio_eos positions: {audio_eos_positions}")
        print(
            f"  unmasked labels span: [{unmasked_start} .. {unmasked_end}] "
            f"({(unmasked_end - unmasked_start + 1) if unmasked else 0} tokens)"
        )
        print(f"  first unmasked token id: {first_unmasked_id} ({tok.decode([first_unmasked_id]) if first_unmasked_id is not None else None!r})")
        print(f"  unmasked decoded: {unmasked_decoded[:300]!r}")
        # Last 16 ids of the row (where the response should be)
        print(f"  ids[-16:]: {ids[-16:].tolist()}")
        print(f"  decoded ids[-16:]: {tok.decode(ids[-16:])!r}")

        # Check for the bug: does the unmasked region START before the response actually starts?
        # The response starts after the audio_eos + "Please describe..." sequence.
        # If unmasked_start <= audio_eos_position, that's clearly wrong (prompt unmasked).
        if audio_eos_positions and unmasked_start is not None:
            last_audio_eos = audio_eos_positions[-1]
            # Prompt continues a few tokens after audio_eos for "Please describe and evaluate..."
            # Just check: is unmasked_start at or before last_audio_eos?
            if unmasked_start <= last_audio_eos:
                print(
                    f"  >>> BUG SIGNAL: unmasked region starts at {unmasked_start} "
                    f"but last <|audio_eos|> is at {last_audio_eos} - prompt tokens are being labeled <<<"
                )
                found_bug = True

        print()

    print("=" * 60)
    if found_bug:
        print("CONCLUSION: label-mask misalignment CONFIRMED on at least one row.")
        print("Fix: force tokenizer.padding_side = 'right' in ALLDDPOCollator")
        print("     OR rewrite _build_labels to right-anchor the response slice.")
    else:
        print("Label masks look correct on all rows. Bug is elsewhere.")


if __name__ == "__main__":
    app()
