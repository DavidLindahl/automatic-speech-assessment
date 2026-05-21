"""CPU probe: where does the supervised label region actually start?

Loads ALLDDPOCollator on 2 real DPO samples and dumps, per row:
  - full input length, prompt-only length, the prompt_len the collator computes
  - the position of the first non-(-100) label
  - the first 8 token IDs from that position, decoded

If the first supervised token is <|AUDIO|>, "Please", or <|audio_eos|>, the
label mask is supervising prompt tokens as response: that is the EOS-collapse
cause. If it is real response text ("This", "The"), the bug is downstream.

CPU only, no generation. Safe on a login node.
"""

import sys

import torch

sys.path.insert(0, "src")
from asa.data import ALLDDPOCollator, DPODataset  # noqa: E402
from transformers import AutoProcessor, AutoTokenizer  # noqa: E402

POLICY_MODEL = "/work3/s234817/automatic-speech-assessment/models/sft_warmup_paper_half_h100"
REF_MODEL = "Qwen/Qwen2-7B"
DPO_JSON = "data/processed/train_dpo_paper_half_h100_clean.json"
DATA_ROOT = "data"


def main() -> None:
    print("Loading processors...")
    audio_processor = AutoProcessor.from_pretrained(POLICY_MODEL)
    text_tokenizer = AutoTokenizer.from_pretrained(REF_MODEL)

    ds = DPODataset(DPO_JSON, DATA_ROOT, max_samples=2)
    features = [ds[0], ds[1]]

    collator = ALLDDPOCollator(audio_processor, text_tokenizer)
    batch = collator(features)

    tok = audio_processor.tokenizer
    input_ids = batch["policy_input_ids"]
    labels = batch["policy_labels"]
    attn = batch["policy_attention_mask"]

    n = len(features)  # first n rows are chosen, next n are rejected
    print(f"\npolicy_input_ids shape : {tuple(input_ids.shape)}")
    print(f"policy_labels shape    : {tuple(labels.shape)}")

    for i in range(input_ids.shape[0]):
        kind = "chosen" if i < n else "rejected"
        real_len = int(attn[i].sum())
        supervised = (labels[i] != -100).nonzero().flatten()
        if len(supervised) == 0:
            print(f"\n--- row {i} ({kind}): NO supervised tokens ---")
            continue
        first = int(supervised[0])
        last = int(supervised[-1])
        n_sup = len(supervised)
        first_ids = input_ids[i, first : first + 8].tolist()
        decoded = tok.decode(input_ids[i, first : first + 8])
        print(f"\n--- row {i} ({kind}) ---")
        print(f"  real (non-pad) length      : {real_len}")
        print(f"  supervised tokens          : {n_sup}")
        print(f"  first supervised position  : {first}")
        print(f"  last supervised position   : {last}")
        print(f"  first 8 supervised ids     : {first_ids}")
        print(f"  first 8 supervised decoded : {decoded!r}")
        # context: 3 tokens before the supervised region
        ctx_start = max(0, first - 3)
        ctx = tok.decode(input_ids[i, ctx_start:first])
        print(f"  3 tokens BEFORE supervised : {ctx!r}")


if __name__ == "__main__":
    main()
