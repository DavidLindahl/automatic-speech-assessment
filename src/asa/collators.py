"""Batch collators for Qwen2-Audio SFT and ALLD-DPO training."""

from typing import Any, Dict, List

import torch

from asa.audio import TARGET_SR


class Qwen2AudioCollator:
    """
    Collates a list of SFTDataset samples into a batch for the trainer.

    1. Concatenates prompt + response into a single text string per sample
    2. Calls processor(text=..., audios=...) to get input_ids + audio features
    3. Creates labels: prompt tokens masked with -100, response tokens kept
    """

    def __init__(self, processor):
        self.processor = processor

    def _prepare_inputs(self, features):
        prompts = [f["prompt"] for f in features]

        eos_token = self.processor.tokenizer.eos_token
        full_texts = [f["prompt"] + f["response"] + eos_token for f in features]
        audios = [f["audio"] for f in features]
        return prompts, full_texts, audios

    def _build_labels(self, batch, prompt_batch, features):
        """Mask prompt and padding tokens with -100 in labels."""
        labels = batch["input_ids"].clone()
        for i in range(len(features)):
            prompt_len = (
                prompt_batch["input_ids"][i]
                .ne(self.processor.tokenizer.pad_token_id)
                .sum()
            )
            labels[i, :prompt_len] = -100
        labels[batch["attention_mask"] == 0] = -100
        return labels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts, full_texts, audios = self._prepare_inputs(features)

        batch = self.processor(
            text=full_texts,
            audios=audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        prompt_batch = self.processor(
            text=prompts,
            audios=audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        batch["labels"] = self._build_labels(batch, prompt_batch, features)
        return batch


class ALLDDPOCollator:
    """
    Dual-stream Collator for the ALLD method.
    Processes audio + text for the Policy Model.
    Processes text-only metadata for the Reference Model.
    """

    def __init__(self, audio_processor, text_tokenizer):
        self.audio_processor = audio_processor
        self.text_tokenizer = text_tokenizer

        # Force right-padding on both tokenizers. _build_labels assumes the real
        # prompt+response starts at index 0; under left-padding (the Qwen2-Audio
        # processor default) the prompt starts after a variable run of PAD
        # tokens, so labels[:prompt_len] masks PADs and leaves the real prompt
        # supervised as response. Confirmed root cause of the DPO collapse
        # (diagnostic 28376116). Right-padding makes the per-row prompt length
        # an exact prefix mask. Reverting this re-introduces the collapse.
        self.audio_processor.tokenizer.padding_side = "right"
        self.text_tokenizer.padding_side = "right"

        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

    def _build_labels(self, batch, prompt_lengths):
        """Mask prompt and padding tokens with -100 in labels.

        prompt_lengths: per-row count of real (non-pad) prompt tokens, computed
        from the prompt-only attention_mask. With right-padding the prompt is an
        exact prefix, so labels[:prompt_len] cleanly masks only the prompt. The
        trailing PAD is masked separately via the full batch attention_mask,
        which is robust even when pad_token_id == eos_token_id (true for the
        Qwen2-7B reference model) since it keys on the mask, not the token id.
        """
        labels = batch["input_ids"].clone()
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, :prompt_len] = -100
        labels[batch["attention_mask"] == 0] = -100
        return labels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}

        # STREAM A: Policy Model (Audio + Text)
        audio_eos = self.audio_processor.tokenizer.eos_token

        audio_prompts = [f["audio_prompt"] for f in features]
        policy_chosen = [f["audio_prompt"] + f["chosen"] + audio_eos for f in features]
        policy_rejected = [
            f["audio_prompt"] + f["rejected"] + audio_eos for f in features
        ]
        audios = [f["audio"] for f in features]

        # 2N Batching for DeepSpeed
        policy_texts = policy_chosen + policy_rejected
        policy_prompts = audio_prompts + audio_prompts
        concat_audios = audios + audios

        policy_inputs = self.audio_processor(
            text=policy_texts,
            audios=concat_audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        policy_prompt_inputs = self.audio_processor(
            text=policy_prompts,
            audios=concat_audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        batch["policy_input_ids"] = policy_inputs["input_ids"]
        batch["policy_attention_mask"] = policy_inputs["attention_mask"]
        batch["policy_audio_values"] = policy_inputs.get("audio_values", None)
        batch["policy_audio_features"] = policy_inputs.get("audio_features", None)
        # Prompt length per row = count of real tokens in the prompt-only batch.
        # Derived from attention_mask, not token-id != pad, so it is correct
        # even when pad_token_id collides with a content token.
        policy_prompt_lens = policy_prompt_inputs["attention_mask"].sum(dim=1)
        batch["policy_labels"] = self._build_labels(
            policy_inputs, policy_prompt_lens
        )

        # STREAM B: Reference Model (Text Only)
        text_eos = self.text_tokenizer.eos_token

        meta_prompts = [f["meta_prompt"] for f in features]
        ref_chosen = [f["meta_prompt"] + f["chosen"] + text_eos for f in features]
        ref_rejected = [f["meta_prompt"] + f["rejected"] + text_eos for f in features]

        # 2N Batching for DeepSpeed
        ref_texts = ref_chosen + ref_rejected
        concat_meta_prompts = meta_prompts + meta_prompts

        ref_inputs = self.text_tokenizer(
            ref_texts,
            return_tensors="pt",
            padding=True,
        )

        ref_prompt_inputs = self.text_tokenizer(
            concat_meta_prompts,
            return_tensors="pt",
            padding=True,
        )

        batch["ref_input_ids"] = ref_inputs["input_ids"]
        batch["ref_attention_mask"] = ref_inputs["attention_mask"]
        ref_prompt_lens = ref_prompt_inputs["attention_mask"].sum(dim=1)
        batch["ref_labels"] = self._build_labels(ref_inputs, ref_prompt_lens)

        return batch
