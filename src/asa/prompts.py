"""Prompt templates and builders shared across SFT, DPO, and inference.

Layout: three TASKS (MOS, MOS+discontinuity, temporal) each needing prompts for
up to three ROLES:
  - query     : the bare instruction the fine-tuned model sees (PROMPT_TEMPLATE).
  - zero-shot : non-leaking instruction for the untrained Instruct baseline.
  - expert    : ALLD reference oracle; leaks the ground-truth scores so the
                frozen text model has a strong per-example logprob to score against.
"""

from asa.audio import AUDIO_SPECIAL


# ============================================================================
# QUERY PROMPT (shared) -- what every fine-tuned checkpoint is trained/queried on
# ============================================================================

# Trailing "\n" is a required prompt/response delimiter: without it Qwen BPE
# merges the prompt tail with the first response token ("speech.This"), masking
# response position 0 and driving DPO EOS-collapse. "speech.\nThis" splits clean.
PROMPT_TEMPLATE = f"{AUDIO_SPECIAL}Please describe and evaluate the synthetic speech.\n"


# ============================================================================
# TASK 1 -- MOS (4 dims: mos / noi / col / loud). The paper's released setup.
# ============================================================================

# Shared 1-5 rubric for the four dimensions. Reused by the MOS expert, both
# zero-shot prompts, and the temporal expert.
DIMENSION_DEFINITIONS_MOS = """I will give you a tuple of meta information for speech quality evaluation, it contains 4 factors are
rating from 1 to 5. For all these factors, higher is better.
    (1) mos: the overall quality. 1 is very bad, 2 is poor, 3 is fair, 4 is good, 5 is excellent.
    (2) noi: the level of noise in the audio, reflecting the impact of background noise or other non-speech interference on audio quality. 1 is very noisy, 2 is somewhat noisy, 3 is neither noisy nor clean, 4 is somewhat clean, and 5 is completely clean.
    (3) col: the alterations in the natural sound of speech caused by distortions or unwanted modifications. 1 is severely distorted, 2 is significantly distorted, 3 is moderately distorted, 4 is slightly distorted, and 5 is no distortion.
    (4) loud: the perceived volume or loudness of the audio. 1 is extremely quiet, 2 is significantly quiet, 3 is soft but understandable, 4 is clearly loud, and 5 is perfectly loud.
"""

# --- MOS expert (reference oracle) blocks -> build_expert_prompt_MOS ---
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


def build_expert_prompt_MOS(mos: float, noi: float, col: float, loud: float) -> str:
    """MOS reference-stream prompt: rubric + task + examples + leaked score tuple.

    The trailing "Output:\\n" is the reference-stream analogue of the
    PROMPT_TEMPLATE "\\n" fix: it keeps the prompt/response boundary a clean BPE
    split so the reference and policy streams align at token position 0.
    """
    current_input = f"\n--- Current Task ---\nInput: {{mos: {mos}, noi: {noi}, col: {col}, loud: {loud}}}\nOutput:\n"
    return (
        DIMENSION_DEFINITIONS_MOS
        + EXPERT_TASK_MOS
        + EXPERT_FEW_SHOT_EXAMPLES_MOS
        + current_input
    )


# ============================================================================
# TASK 2 -- MOS + discontinuity (5 dims, adds `dis`). Opt-in ablation only.
# Enabled via DPODataset(use_discontinuity=True); the 4-dim path is unchanged.
# ============================================================================

# --- MOS+dis expert blocks -> build_expert_prompt_MOS_DIS ---
DIMENSION_DEFINITIONS_MOS_DIS = """I will give you a tuple of meta information for speech quality evaluation, it contains 5 factors are
rating from 1 to 5. For all these factors, higher is better.
    (1) mos: the overall quality. 1 is very bad, 2 is poor, 3 is fair, 4 is good, 5 is excellent.
    (2) noi: the level of noise in the audio, reflecting the impact of background noise or other non-speech interference on audio quality. 1 is very noisy, 2 is somewhat noisy, 3 is neither noisy nor clean, 4 is somewhat clean, and 5 is completely clean.
    (3) col: the alterations in the natural sound of speech caused by distortions or unwanted modifications. 1 is severely distorted, 2 is significantly distorted, 3 is moderately distorted, 4 is slightly distorted, and 5 is no distortion.
    (4) dis: the continuity of the speech, reflecting interruptions, dropouts, or other breaks in the signal. 1 is severely discontinuous, 2 is significantly discontinuous, 3 is moderately discontinuous, 4 is slightly discontinuous, and 5 is perfectly continuous.
    (5) loud: the perceived volume or loudness of the audio. 1 is extremely quiet, 2 is significantly quiet, 3 is soft but understandable, 4 is clearly loud, and 5 is perfectly loud.
"""

EXPERT_TASK_MOS_DIS = """I need you to generate a descriptive evaluation for this speech, including a description according to
the score from noise, coloration, discontinuity, and loudness, analyze how they influence the overall quality, and add the mos in the end.
"""

EXPERT_FEW_SHOT_EXAMPLES_MOS_DIS = """
--- Example 1 ---
Input: {mos: 4.5, noi: 5.0, col: 4.5, dis: 4.8, loud: 4.8}
Output: This speech is highly intelligible and perfectly loud. There is no background noise, the speech is continuous, and there is only a very slight coloration that is barely noticeable. Taking all factors into account, the overall MOS is 4.5.

--- Example 2 ---
Input: {mos: 2.1, noi: 3.0, col: 2.5, dis: 2.4, loud: 4.0}
Output: The volume of the speech is clear and adequately loud. However, there is moderate background noise, noticeable distortion, and the speech is frequently discontinuous. These degradations make the speech sound unnatural overall, so the MOS score is only 2.1.
"""


def build_expert_prompt_MOS_DIS(
    mos: float, noi: float, col: float, dis: float, loud: float
) -> str:
    """5-dim MOS reference prompt with discontinuity re-added.

    Mirrors build_expert_prompt_MOS exactly (including the "Output:\\n" fix); the
    only difference is the extra `dis` score. Used only by the discontinuity
    ablation to test whether the dropped dimension changes global ALLD.
    """
    current_input = (
        f"\n--- Current Task ---\nInput: {{mos: {mos}, noi: {noi}, "
        f"col: {col}, dis: {dis}, loud: {loud}}}\nOutput:\n"
    )
    return (
        DIMENSION_DEFINITIONS_MOS_DIS
        + EXPERT_TASK_MOS_DIS
        + EXPERT_FEW_SHOT_EXAMPLES_MOS_DIS
        + current_input
    )


# ============================================================================
# TASK 3 -- Temporal (MOS + degradation interval). Our extension.
# ============================================================================

# --- Temporal expert (reference oracle) blocks -> build_expert_prompt_TEMPORAL ---
# Oracle: leaks scores AND the ground-truth start/end. Few-shot Outputs use the
# free-text "between START and END" form (the trained target format, not the
# <aN><fK> tokens) so the reference scores chosen/rejected text on-distribution.
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
    """Temporal reference-stream prompt: rubric + temporal task + examples + leaked
    scores and interval.

    Reuses the 4-dim DIMENSION_DEFINITIONS_MOS (the released caption set is 4-dim)
    and keeps the "Output:\\n" BPE-boundary fix, matching build_expert_prompt_MOS.
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


# ============================================================================
# ZERO-SHOT BASELINES -- untrained Qwen2-Audio-7B-Instruct, the "before" floor.
# Rendered through the model's ChatML chat template (it descends from Instruct),
# not the bare PROMPT_TEMPLATE. Comparability comes from the shared metric code
# in evaluate.py / evaluate_temporal.py, not from an identical prompt string.
# Both are strictly non-leaking: rubric + ask, no ground truth, no examples.
# ============================================================================

# --- Zero-shot MOS ---
ZEROSHOT_TASK_MOS = (
    "I need you to generate a descriptive evaluation for this speech, "
    "including a description according to its noise, coloration, and loudness, "
    "analyze how they influence the overall quality, and add the overall MOS "
    "score (a number from 1 to 5) at the end."
)

ZEROSHOT_USER_TEXT_MOS = DIMENSION_DEFINITIONS_MOS + ZEROSHOT_TASK_MOS

# --- Zero-shot temporal ---
# Format ask uses a NEUTRAL placeholder ("X and Y"), never a concrete example:
# an early smoke that showed "between 1.2 and 3.4 seconds" made the model parrot
# that exact interval on every sample, faking a chance-overlap t-IoU. The phrase
# "between X and Y seconds" is chosen to hit the `range` regex in
# evaluate_temporal.extract_interval (the honest parse path); we don't ask for
# our <|float|> / <aN><fK> tokens since the untrained model doesn't know them.
ZEROSHOT_TASK_TEMPORAL = (
    "Listen to this speech clip. Part of it is degraded in quality while the "
    "rest is clean. First, briefly describe the speech quality and its overall "
    "MOS score (a number from 1 to 5). Then identify the single span of time "
    "where the degradation occurs, and state it explicitly as a time range in "
    "seconds in the form \"between X and Y seconds\", where X and Y are the "
    "start and end times you identify."
)

ZEROSHOT_USER_TEXT_TEMPORAL = DIMENSION_DEFINITIONS_MOS + ZEROSHOT_TASK_TEMPORAL


def build_zeroshot_prompt_MOS(processor) -> str:
    """Render the zero-shot MOS baseline prompt through the ChatML chat template.

    Off-the-shelf audio LLMs cannot do this task zero-shot (Chen et al. 2501.17202,
    Sec. 4 / App. B); this row reproduces that finding. The audio placeholder
    expands to the standard <|audio_bos|><|AUDIO|><|audio_eos|> block, so the
    result flows through asa.inference.run_inference unchanged.
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


def build_zeroshot_prompt_temporal(processor) -> str:
    """Render the zero-shot temporal baseline prompt through the ChatML chat template.

    Temporal counterpart to build_zeroshot_prompt_MOS: the defensible "before"
    floor for the temporal results. Same single-audio-token rendering, so it runs
    through asa.inference.run_inference unchanged.
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
