"""Prompt templates and expert-prompt builders shared across SFT, DPO, and inference."""

from asa.audio import AUDIO_SPECIAL


# The trailing newline is a deliberate prompt/response delimiter. Without it,
# Qwen BPE merges the prompt tail with the first response word ("speech.This"
# -> ".This" as one token), so the prompt-length label mask hides the first
# response token and the model is never trained to produce position 0 at
# inference. That distribution shift drives the DPO EOS-collapse (the model
# defaults to <|im_end|> at the first generated position). The "\n" breaks the
# merge: "speech.\nThis" tokenizes with "This" as a clean standalone token.
PROMPT_TEMPLATE = (
    f"{AUDIO_SPECIAL}Please describe and evaluate the synthetic speech.\n"
)


DIMENSION_DEFINITIONS_MOS = """I will give you a tuple of meta information for speech quality evaluation, it contains 4 factors are
rating from 1 to 5. For all these factors, higher is better.
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
