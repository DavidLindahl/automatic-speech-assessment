"""
prompts.py - Prompt generation templates for ALLD expert models.
"""

# --- ALLD Expert Text Prompt Templates ---
# ==========================================
# 1. SINGLE MOS PROMPTS (No 'dis' feature)
# ==========================================

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
    current_input = f"\n--- Current Task ---\nInput: {{mos: {mos}, noi: {noi}, col: {col}, loud: {loud}}}\nOutput:"
    return (
        DIMENSION_DEFINITIONS_MOS
        + EXPERT_TASK_MOS
        + EXPERT_FEW_SHOT_EXAMPLES_MOS
        + current_input
    )


# ==========================================
# 2. A/B TEST PROMPTS (Includes 'dis' feature)
# ==========================================

DIMENSION_DEFINITIONS_AB = """I will give you a tuple of meta information for speech quality evaluation, it contains 5 factors are
rating from 1 to 5. For all these factors, higher is better.
    (1) mos: the overall quality. 1 is very bad, 2 is poor, 3 is fair, 4 is good, 5 is excellent.
    (2) noi: the level of noise in the audio, reflecting the impact of background noise or other non-speech interference on audio quality. 1 is very noisy, 2 is somewhat noisy, 3 is neither noisy nor clean, 4 is somewhat clean, and 5 is completely clean.
    (3) col: the alterations in the natural sound of speech caused by distortions or unwanted modifications. 1 is severely distorted, 2 is significantly distorted, 3 is moderately distorted, 4 is slightly distorted, and 5 is no distortion.
    (4) dis: the discontinuity in the audio, reflecting whether there are breaks, stutters, or incoherence during playback. 1 is severely discontinuous, 2 is significantly discontinuous, 3 is moderately discontinuous, 4 is slightly discontinuous, and 5 is no discontinuity.
    (5) loud: the perceived volume or loudness of the audio. 1 is extremely quiet, 2 is significantly quiet, 3 is soft but understandable, 4 is clearly loud, and 5 is perfectly loud.
"""

EXPERT_TASK_AB = """I need you to perform A/B test according to their mos (mos higher means winner). You can flexibly
select 1 to 3 aspects from the sub-dimensions with an obvious gap (usually score difference more than 0.5), then
compare them according to these distinctions. Finally, please give your preference with a reasonable
analysis.
"""

EXPERT_FEW_SHOT_EXAMPLES_AB = """
--- Example 1 ---
Input: {A_mos: 1.8, A_noi: 3.2, A_col: 1.9, A_dis: 2.3, A_loud: 3.3, B_mos: 3.6, B_noi: 2.6, B_col: 2.8, B_dis: 4.0, B_loud: 3.7}
Output: SpeechA and SpeechB have similar levels of distortion and loudness. However, SpeechB has better continuity than SpeechA. Although SpeechA is slightly cleaner, I would select SpeechB as better synthesized speech due to its significantly higher overall synthetic quality and better continuity.

--- Example 2 ---
Input: {A_mos: 4.0, A_noi: 4.2, A_col: 3.7, A_dis: 3.9, A_loud: 4.1, B_mos: 1.6, B_noi: 2.4, B_col: 1.8, B_dis: 2.5, B_loud: 1.6}
Output: SpeechA and SpeechB have significant gaps in several aspects. SpeechA has much lower noise, less distortion, and better continuity than SpeechB. Additionally, SpeechA is also much louder than SpeechB. Considering these substantial differences, I would select SpeechA as the better synthesized speech.
"""


def build_expert_prompt_ab(
    A_mos: float,
    A_noi: float,
    A_col: float,
    A_dis: float,
    A_loud: float,
    B_mos: float,
    B_noi: float,
    B_col: float,
    B_dis: float,
    B_loud: float,
) -> str:
    current_input = f"\n--- Current Task ---\nInput: {{A_mos: {A_mos}, A_noi: {A_noi}, A_col: {A_col}, A_dis: {A_dis}, A_loud: {A_loud}, B_mos: {B_mos}, B_noi: {B_noi}, B_col: {B_col}, B_dis: {B_dis}, B_loud: {B_loud}}}\nOutput:"
    return (
        DIMENSION_DEFINITIONS_AB
        + EXPERT_TASK_AB
        + EXPERT_FEW_SHOT_EXAMPLES_AB
        + current_input
    )
