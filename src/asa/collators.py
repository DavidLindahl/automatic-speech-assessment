"""
collators.py - Collation functions for PyTorch dataloaders.
"""

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

        # ADD THE EOS TOKEN HERE:
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
            audio=audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        prompt_batch = self.processor(
            text=prompts,
            audio=audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        batch["labels"] = self._build_labels(batch, prompt_batch, features)
        return batch


class Qwen2AudioCollatorAB(Qwen2AudioCollator):
    """
    A/B preference variant of Qwen2AudioCollator.

    Each sample contributes *two* audios.  The Qwen2-Audio processor expects a
    **flat** list of waveforms — it assigns them to ``<|AUDIO|>`` tokens
    sequentially across the batch.
    """

    def _prepare_inputs(self, features):
        """Return (prompts, full_texts, audios) — audios flattened."""
        prompts = [f["prompt"] for f in features]

        # ADD THE EOS TOKEN HERE FOR A/B SFT:
        eos_token = self.processor.tokenizer.eos_token
        full_texts = [f["prompt"] + f["response"] + eos_token for f in features]

        # Flat list: [sample0_a, sample0_b, sample1_a, sample1_b, ...]
        audios = [audio for f in features for audio in [f["audio_a"], f["audio_b"]]]

        return prompts, full_texts, audios


class ALLDDPOCollator:
    """
    Dual-stream Collator for the ALLD method.
    Processes audio + text for the Policy Model.
    Processes text-only metadata for the Reference Model.
    """

    def __init__(self, audio_processor, text_tokenizer):
        self.audio_processor = audio_processor
        self.text_tokenizer = text_tokenizer

        # Ensure text tokenizer has a pad token
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

    def _build_labels(self, batch, prompt_batch, tokenizer):
        """Mask prompt and padding tokens with -100 in labels."""
        labels = batch["input_ids"].clone()
        for i in range(len(prompt_batch["input_ids"])):
            prompt_len = prompt_batch["input_ids"][i].ne(tokenizer.pad_token_id).sum()
            labels[i, :prompt_len] = -100
        labels[batch["attention_mask"] == 0] = -100
        return labels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}

        # ==========================================
        # 1. STREAM A: Policy Model (Audio + Text)
        # ==========================================

        # GET THE EOS TOKEN FOR THE POLICY MODEL
        audio_eos = self.audio_processor.tokenizer.eos_token

        audio_prompts = [f["audio_prompt"] for f in features]

        # APPEND EOS TOKEN TO THE RESPONSES
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
            audio=concat_audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        policy_prompt_inputs = self.audio_processor(
            text=policy_prompts,
            audio=concat_audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        batch["policy_input_ids"] = policy_inputs["input_ids"]
        batch["policy_attention_mask"] = policy_inputs["attention_mask"]
        batch["policy_audio_values"] = policy_inputs.get(
            "audio_values", None
        )  # Handle processor variations
        batch["policy_audio_features"] = policy_inputs.get("audio_features", None)
        batch["policy_labels"] = self._build_labels(
            policy_inputs, policy_prompt_inputs, self.audio_processor.tokenizer
        )

        # ==========================================
        # 2. STREAM B: Reference Model (Text Only)
        # ==========================================

        # GET THE EOS TOKEN FOR THE REFERENCE MODEL
        text_eos = self.text_tokenizer.eos_token

        meta_prompts = [f["meta_prompt"] for f in features]

        # APPEND EOS TOKEN TO THE RESPONSES
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
        batch["ref_labels"] = self._build_labels(
            ref_inputs, ref_prompt_inputs, self.text_tokenizer
        )

        return batch


class ALLDDPOCollatorAB(ALLDDPOCollator):
    """
    A/B preference variant of ALLDDPOCollator.
    The primary difference is flattening the dual `audio_a` and `audio_b` into
    a single sequence so Qwen2-Audio's processor correctly maps them to the
    two <|AUDIO|> tags in the prompt.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}

        # ==========================================
        # 1. STREAM A: Policy Model (Audio + Text)
        # ==========================================
        audio_eos = self.audio_processor.tokenizer.eos_token

        audio_prompts = [f["audio_prompt"] for f in features]

        # FIX: Append EOS to chosen/rejected
        policy_chosen = [f["audio_prompt"] + f["chosen"] + audio_eos for f in features]
        policy_rejected = [
            f["audio_prompt"] + f["rejected"] + audio_eos for f in features
        ]

        # Flatten audios for AB tests: [sample0_a, sample0_b, sample1_a, sample1_b, ...]
        audios = [audio for f in features for audio in [f["audio_a"], f["audio_b"]]]

        # 2N Batching for DeepSpeed
        policy_texts = policy_chosen + policy_rejected
        policy_prompts = audio_prompts + audio_prompts
        concat_audios = audios + audios

        policy_inputs = self.audio_processor(
            text=policy_texts,
            audio=concat_audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        policy_prompt_inputs = self.audio_processor(
            text=policy_prompts,
            audio=concat_audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )

        batch["policy_input_ids"] = policy_inputs["input_ids"]
        batch["policy_attention_mask"] = policy_inputs["attention_mask"]
        batch["policy_audio_values"] = policy_inputs.get("audio_values", None)
        batch["policy_audio_features"] = policy_inputs.get("audio_features", None)
        batch["policy_labels"] = self._build_labels(
            policy_inputs, policy_prompt_inputs, self.audio_processor.tokenizer
        )

        # ==========================================
        # 2. STREAM B: Reference Model (Text Only)
        # ==========================================
        text_eos = self.text_tokenizer.eos_token

        meta_prompts = [f["meta_prompt"] for f in features]

        # FIX: Append EOS to chosen/rejected
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
        batch["ref_labels"] = self._build_labels(
            ref_inputs, ref_prompt_inputs, self.text_tokenizer
        )

        return batch
