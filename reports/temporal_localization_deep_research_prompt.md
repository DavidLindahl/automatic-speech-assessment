# Deep Research Prompt: Temporal Speech Degradation Localization

I am working on a bachelor project about automatic speech assessment with Qwen2-Audio. We fine-tuned a model to describe synthetic speech quality and localize a degraded audio region by generating timestamp tokens like `<|2.88|>` and `<|4.73|>`.

## Current Model And Evaluation

- Model evaluated: `Leng2beat/speech-quality-assessement-qwen2audio-sft-temporal-max-mos3-partial-step305`
- Dataset: `data/processed/train_nisqa_temporal_mix_max_mos3.json`
- Result path: `results/evaluation/sft_temporal_max_mos3_h100_fixed/train_nisqa_temporal_mix_max_mos3_results.json`
- Evaluation script: `src/asa/evaluate_temporal.py`
- Job script family: `jobs/evaluate/evaluate_sft_temporal_max_mos3_*.sh`
- Prompt mode: query prompt
- Decoding: greedy, `do_sample=false`
- Max new tokens: 150
- Timestamp extraction: preserve generated special tokens with `skip_special_tokens=False`, then parse `<|number|>` token pairs.

Important context: an earlier evaluation accidentally decoded with `skip_special_tokens=True`, which stripped timestamp tokens and produced outputs like `between  and .`. That bug is fixed. The current run successfully parses timestamps for all samples.

## Current Metrics

```json
{
  "samples_total": 5136,
  "samples_with_ground_truth_interval": 5136,
  "samples_with_parsed_prediction_interval": 5136,
  "mean_tiou": 0.14086944006108137,
  "median_tiou": 0.0,
  "hit_iou_ge_0_1": 0.3099688473520249,
  "hit_iou_ge_0_3": 0.20989096573208724,
  "hit_iou_ge_0_5": 0.125,
  "mean_start_abs_err": 2.634770054517134,
  "mean_end_abs_err": 2.6923637071651094,
  "prompt_mode": "query",
  "do_sample": false,
  "temperature": 0.7,
  "top_p": 0.9,
  "max_new_tokens": 150
}
```

Note: temperature is recorded as `0.7` by the CLI defaults, but greedy decoding is used, so temperature is not actually used by Hugging Face generation. A follow-up H100 job will explicitly pass `--temperature 0.0` so the output metadata is less confusing.

## Example Predictions

Ground truth:

```text
The quality is interrupted by background noise, codec artifacts, and packet-loss concealment artifacts occurring between <|2.88|> and <|4.73|>.
```

Prediction:

```text
The quality is interrupted by codec artifacts and packet-loss concealment artifacts occurring between <|0.27|> and <|1.99|>.
```

Ground truth:

```text
The quality is interrupted by codec artifacts and packet-loss concealment artifacts occurring between <|6.96|> and <|9.76|>.
```

Prediction:

```text
The quality is interrupted by codec artifacts and packet-loss concealment artifacts occurring between <|7.01|> and <|8.71|>.
```

Observed behavior:

- The model now always emits parseable timestamp tokens.
- Localization is weak: median t-IoU is 0.0 and Hit@0.5 is 12.5%.
- Some examples are good partial overlaps, but many predictions are far from the ground truth.
- The text often collapses to a generic quality description and repeated degradation labels.
- Some timestamp pairs repeat across unrelated samples, for example `<|7.01|>` to `<|8.71|>`.
- This evaluation is on the temporal training JSON, so weak training-set localization suggests the model has not learned the temporal task well, or the training/evaluation setup is misaligned.

## Research Questions

Please investigate and propose concrete next steps for improving temporal localization in an audio-language model fine-tuned on synthetic degradation intervals.

Focus on:

1. Whether timestamp tokens like `<|2.88|>` should be treated as special tokens, regular text tokens, discretized bins, or a separate regression head.
2. Whether Qwen2-Audio style generative decoding is a good fit for second-level temporal localization.
3. Training recipe issues that could cause generic text plus weak timestamp learning, including too little temporal data, loss imbalance between prose and timestamps, tokenization problems, prompt mismatch, catastrophic forgetting, or insufficient training steps.
4. Better evaluation splits and baselines, since the current run is on `train_nisqa_temporal_mix_max_mos3.json`.
5. Practical model-improvement experiments ranked by expected impact and cost.
6. Whether the synthetic mixing procedure and one-interval supervision are likely to produce learnable timestamp cues.

Please produce:

- A diagnosis of the most likely failure modes.
- A prioritized experiment plan.
- Concrete changes to data format, prompts, decoding, and training.
- Suggested metrics and plots beyond mean t-IoU.
- Any relevant papers or methods for audio event localization, temporal grounding, or timestamp-token training in multimodal LLMs.

## Follow-Up Job

We are rerunning the same evaluation on H100 with explicit temperature metadata:

```bash
bsub < jobs/evaluate/evaluate_sft_temporal_max_mos3_h100_temp0.sh
```

Expected new result path:

```text
results/evaluation/sft_temporal_max_mos3_h100_temp0/train_nisqa_temporal_mix_max_mos3_results.json
```
