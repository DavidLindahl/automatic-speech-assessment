"""Prompt templates and expert-prompt builders shared across SFT, DPO, and inference."""

from asa.audio import AUDIO_SPECIAL


# The trailing newline is a deliberate prompt/response delimiter. Without it,
# Qwen BPE merges the prompt tail with the first response word ("speech.This"
# -> ".This" as one token), so the prompt-length label mask hides the first
# response token and the model is never trained to produce position 0 at
# inference. That distribution shift drives the DPO EOS-collapse (the model
# defaults to <|im_end|> at the first generated position). The "\n" breaks the
# merge: "speech.\nThis" tokenizes with "This" as a clean standalone token.
PROMPT_TEMPLATE = f"{AUDIO_SPECIAL}Please describe and evaluate the synthetic speech.\n"


DIMENSION_DEFINITIONS_MOS = """I will give you a tuple of meta information for speech quality evaluation, it contains 4 factors are
rating from 1 to 5. For all these factors, higher is better.
    (1) mos: the overall quality. 1 is very bad, 2 is poor, 3 is fair, 4 is good, 5 is excellent.
    (2) noi: the level of noise in the audio, reflecting the impact of background noise or other non-speech interference on audio quality. 1 is very noisy, 2 is somewhat noisy, 3 is neither noisy nor clean, 4 is somewhat clean, and 5 is completely clean.
    (3) col: the alterations in the natural sound of speech caused by distortions or unwanted modifications. 1 is severely distorted, 2 is significantly distorted, 3 is moderately distorted, 4 is slightly distorted, and 5 is no distortion.
    (4) loud: the perceived volume or loudness of the audio. 1 is extremely quiet, 2 is significantly quiet, 3 is soft but understandable, 4 is clearly loud, and 5 is perfectly loud.
"""

DIMENSION_DEFINITIONS_ZEROSHOT = """Here are the definitions of the 4 factors for speech quality evaluation, each rated from 1 to 5. For all these factors, higher is better.
    (1) mos: the overall quality. 1 is very bad, 2 is poor, 3 is fair, 4 is good, 5 is excellent.
    (2) noi: the level of noise in the audio, reflecting the impact of background noise or other non-speech interference on audio quality. 1 is very noisy, 2 is somewhat noisy, 3 is neither noisy nor clean, 4 is somewhat clean, and 5 is completely clean.
    (3) col: the alterations in the natural sound of speech caused by distortions or unwanted modifications. 1 is severely distorted, 2 is significantly distorted, 3 is moderately distorted, 4 is slightly distorted, and 5 is no distortion.
    (4) loud: the perceived volume or loudness of the audio. 1 is extremely quiet, 2 is significantly quiet, 3 is soft but understandable, 4 is clearly loud, and 5 is perfectly loud.
"""

EXPERT_TASK_MOS = """I need you to generate a descriptive evaluation for this speech, including a description according to
the score from noise, coloration, and loudness, analyze how they influence the overall quality, and add the mos in the end.
"""

EXPERT_FEW_SHOT_EXAMPLES_MOS = """
--- Example 1 ---
Input: {mos: 4.5, noi: 5.0, col: 4.5, loud: 4.8}
Output: This speech is highly intelligible and perfectly loud. There is no background noise, and there is only a very slight coloration that is barely noticeable. Taking all factors into account, the overall MOS is 4.5.

--- Example 2 ---
Input: {mos: 2.1, noi: 3.0, col: 2.5, loud: 4.0}
Output: The volume of the speech is clear and adequately loud. However, there is moderate background noise and noticeable distortion. These degradations make the speech sound unnatural overall, so the MOS score is only 2.1.
"""


# Non-leaking instruction for the zero-shot baseline: dimension definitions plus
# an explicit "describe and end with an MOS score" ask. The definitions do not
# promise a tuple because the ground-truth (mos, noi, col, loud) tuple is
# deliberately omitted. Promising a tuple here causes instruction-following
# models to invent one instead of assessing the audio. No worked examples are
# included, so the baseline stays strictly zero-shot rather than few-shot.
ZEROSHOT_TASK_MOS = (
    "I need you to generate a descriptive evaluation for this speech, "
    "including a description according to its noise, coloration, and loudness, "
    "analyze how they influence the overall quality, and add the overall MOS "
    "score (a number from 1 to 5) at the end."
)

ZEROSHOT_USER_TEXT_MOS = DIMENSION_DEFINITIONS_ZEROSHOT + ZEROSHOT_TASK_MOS


def build_zeroshot_prompt_MOS(processor) -> str:
    """Instructed zero-shot prompt for the off-the-shelf (untrained) baseline.

    This measures what Qwen2-Audio can do on speech quality assessment *before*
    any fine-tuning. The source paper (Chen et al. 2501.17202, Sec. 4 and
    Appendix B) reports that off-the-shelf audio LLMs, Qwen2-Audio included,
    cannot perform this task zero-shot and tend to hallucinate; this baseline row
    exists to reproduce that finding, not to be competitive.

    The prompt is rendered through the model's **chat template**
    (``processor.apply_chat_template``), not as bare text. The zero-shot baseline
    is ``Qwen2-Audio-7B-Instruct``, which was instruction-tuned on ChatML; the
    fine-tuned ASA checkpoints instead trained on the bare ``PROMPT_TEMPLATE``
    because they descend from the *base* ``Qwen2-Audio-7B``. Feeding the Instruct
    model bare text would be off-distribution and would invite the (correct)
    criticism that the baseline failed only because it was prompted wrong, which
    is exactly the trap to avoid when the whole point is a defensible "before"
    row. Using a different, model-appropriate prompt for the baseline is fine and
    expected: comparability comes from the shared metric code path
    (MAE/MSE/BLEU/ROUGE/BERTScore in ``evaluate.py``), not from an identical
    prompt string.

    The user turn carries the non-leaking instruction (``ZEROSHOT_USER_TEXT_MOS``:
    dimension definitions + "end with an MOS score", **no** ground-truth tuple,
    **no** worked examples) alongside a single audio placeholder. The chat
    template expands that placeholder to the standard
    ``<|audio_bos|><|AUDIO|><|audio_eos|>`` block, so the rendered string still
    flows through the existing :func:`asa.inference.run_inference` unchanged: the
    processor sees exactly one audio token and aligns the audio features just as
    it does for every other eval.

    Args:
        processor: A Qwen2-Audio ``AutoProcessor`` whose tokenizer carries the
            ChatML ``chat_template`` (the Instruct processor does).

    Returns:
        The fully rendered ChatML prompt string, ending in the assistant
        generation prompt and ready to pass as a ``prompt_texts`` entry.
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "placeholder.wav"},
                {"type": "text", "text": ZEROSHOT_USER_TEXT_MOS},
            ],
        }
    ]
    return processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )


# Non-leaking instruction for the zero-shot *temporal* baseline. It asks the
# off-the-shelf model to do the same job the fine-tuned temporal models do
# (describe the speech and say when the degradation occurs), but it never
# reveals the ground-truth interval. The output format ask is deliberately
# phrased as "between X and Y seconds" so a free-text answer lands on the
# `range` regex in evaluate_temporal.extract_interval (between|from ... and|to),
# which is the honest parse path. The untrained Instruct model does not know our
# <|float|> or <aN><fK> timestamp conventions, so we do not ask for them; we ask
# for plain seconds and let the range parser pick them up. A baseline that
# cannot produce a localizable range is an honest low parse rate, not a number
# to manufacture.
#
# CRITICAL (2026-06-09): the format ask uses a NEUTRAL placeholder ("X and Y"),
# never a concrete worked example. The first smoke (28615768) handed the model
# the example "between 1.2 and 3.4 seconds" and the untrained model parroted
# that exact interval on all 20 samples, producing a chance-overlap t-IoU ~0.30
# that would have badly overstated the baseline. Any concrete number here, or
# any directional hint ("near the start"), is contamination: it biases the
# baseline toward a constant answer instead of measuring what the model does on
# its own. Keep the placeholder abstract.
ZEROSHOT_TASK_TEMPORAL = (
    "Listen to this speech clip. Part of it is degraded in quality while the "
    "rest is clean. First, briefly describe the speech quality and its overall "
    "MOS score (a number from 1 to 5). Then identify the single span of time "
    "where the degradation occurs, and state it explicitly as a time range in "
    'seconds in the form "between X and Y seconds", where X and Y are the '
    "start and end times you identify."
)

ZEROSHOT_USER_TEXT_TEMPORAL = DIMENSION_DEFINITIONS_ZEROSHOT + ZEROSHOT_TASK_TEMPORAL


def build_zeroshot_prompt_temporal(processor) -> str:
    """Instructed zero-shot prompt for the off-the-shelf temporal baseline.

    This is the temporal-localization counterpart to
    :func:`build_zeroshot_prompt_MOS`. It measures what Qwen2-Audio can do on
    "when is the degradation" *before* any fine-tuning, the defensible "before"
    floor for the temporal results, and the parser counterpart to the source
    paper's finding that off-the-shelf audio LLMs cannot do this task zero-shot.

    Like the MOS zero-shot prompt, it is rendered through the model's **chat
    template** (``processor.apply_chat_template``), not as bare text, because the
    baseline is ``Qwen2-Audio-7B-Instruct`` (ChatML-tuned) while the fine-tuned
    temporal checkpoints descend from the *base* model and trained on the bare
    query prompt. Feeding the Instruct model bare text would be off-distribution
    and would invite the criticism that the baseline failed only because it was
    prompted wrong. Comparability comes from the shared metric code path
    (t-IoU, Hit@k, start/end error in ``evaluate_temporal.py``), not from an
    identical prompt string.

    The user turn carries the non-leaking instruction
    (``ZEROSHOT_USER_TEXT_TEMPORAL``: dimension definitions + "describe, give an
    MOS, and state the degraded span as a seconds range", **no** ground-truth
    interval, **no** worked examples) alongside a single audio placeholder. The
    chat template expands that placeholder to the standard
    ``<|audio_bos|><|AUDIO|><|audio_eos|>`` block, so the rendered string flows
    through :func:`asa.inference.run_inference` unchanged with exactly one audio
    token.

    Args:
        processor: A Qwen2-Audio ``AutoProcessor`` whose tokenizer carries the
            ChatML ``chat_template`` (the Instruct processor does).

    Returns:
        The fully rendered ChatML prompt string, ending in the assistant
        generation prompt and ready to pass as a ``prompt_texts`` entry.
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "placeholder.wav"},
                {"type": "text", "text": ZEROSHOT_USER_TEXT_TEMPORAL},
            ],
        }
    ]
    return processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )


def build_expert_prompt_MOS(mos: float, noi: float, col: float, loud: float) -> str:
    # The trailing "\n" after "Output:" is the reference-stream analogue of
    # the PROMPT_TEMPLATE "\n" delimiter fix (commit a007248). Without it,
    # "Output:" + "The" merges to a single BPE token "Output:The", so the
    # rejected reference stream's first supervised token becomes
    # " synthesized" instead of "The". That misaligns the DPO reward at
    # position 0: policy sees "The", reference sees " synthesized". The "\n"
    # makes the prompt/response boundary a clean split, identical to the
    # policy stream. Verified by probe_collator_labels.py.
    current_input = f"\n--- Current Task ---\nInput: {{mos: {mos}, noi: {noi}, col: {col}, loud: {loud}}}\nOutput:\n"
    return (
        DIMENSION_DEFINITIONS_MOS
        + EXPERT_TASK_MOS
        + EXPERT_FEW_SHOT_EXAMPLES_MOS
        + current_input
    )


# --- Temporal global-caption reference (ALLD dual-stream, temporal task) ---
# The temporal expert prompt extends the MOS expert with the degradation
# interval. The reference text model is given the full quality palette
# (mos/noi/col/loud) AND the ground-truth start/end, and is asked to produce a
# target in the exact global-caption-temporal form the policy is trained on:
# "The degradation in the clip is between <start> and <end> and <quality caption>".
# Like the MOS expert it is an oracle: it echoes the interval and the scores so
# the reference logprob is a strong per-example baseline. The few-shot Output
# strings match the trained target format (leading temporal clause, MOS in the
# caption) so the reference scores the chosen/rejected text on-distribution.

EXPERT_TASK_TEMPORAL = """I need you to generate a temporal quality evaluation for this speech. Most of the clip is clean; a single span of time carries the degradation. State that span first as "The degradation in the clip is between START and END and", then continue with a description according to the score from noise, coloration, and loudness, analyze how they influence the overall quality, and add the mos in the end.
"""

EXPERT_FEW_SHOT_EXAMPLES_TEMPORAL = """
--- Example 1 ---
Input: {mos: 1.4, noi: 1.4, col: 2.6, loud: 3.0, start: 3.57, end: 4.72}
Output: The degradation in the clip is between 3.57 and 4.72 and is quite noisy and discontinuous, with moderate distortion. Although the loudness is soft but understandable, the overall MOS score is only 1.4.

--- Example 2 ---
Input: {mos: 2.4, noi: 3.0, col: 2.5, loud: 2.0, start: 1.66, end: 3.14}
Output: The degradation in the clip is between 1.66 and 3.14 and is relatively quiet and has some continuity. However, it is somewhat noisy and significantly distorted, which impacts its overall quality. The overall MOS score is about 2.4.
"""


def build_expert_prompt_TEMPORAL(
    mos: float, noi: float, col: float, loud: float, start: float, end: float
) -> str:
    """Build the ALLD reference-stream prompt for the temporal task.

    Args:
        mos, noi, col, loud: The four NISQA quality dimensions (1-5).
        start, end: The ground-truth degradation interval in seconds.

    Returns:
        The text-expert prompt: dimension definitions + temporal task framing +
        temporal few-shot examples + the structured current-task input. The
        trailing ``"\\n"`` after ``Output:`` keeps the prompt/response boundary a
        clean BPE split, matching ``build_expert_prompt_MOS``.
    """
    current_input = (
        "\n--- Current Task ---\n"
        f"Input: {{mos: {mos}, noi: {noi}, col: {col}, loud: {loud}, "
        f"start: {start}, end: {end}}}\nOutput:\n"
    )
    return (
        DIMENSION_DEFINITIONS_MOS
        + EXPERT_TASK_TEMPORAL
        + EXPERT_FEW_SHOT_EXAMPLES_TEMPORAL
        + current_input
    )
