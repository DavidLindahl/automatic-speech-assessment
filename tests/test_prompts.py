import pytest
from asa.prompts import build_expert_prompt_MOS, build_expert_prompt_ab


def test_build_expert_prompt_MOS():
    """Verify the MOS expert prompt builds correctly without discontinued metadata."""
    prompt = build_expert_prompt_MOS(mos=4.5, noi=5.0, col=4.5, loud=4.8)

    # Assert definitions included
    assert "rating from 1 to 5" in prompt
    assert "(1) mos: the overall quality." in prompt
    assert "(2) noi:" in prompt
    assert "(3) col:" in prompt
    assert "(4) loud:" in prompt
    
    # Assert not containing 'dis' in definition list
    assert "(4) dis:" not in prompt
    assert "(5) dis:" not in prompt

    # Assert current input values are populated properly
    assert "Input: {mos: 4.5, noi: 5.0, col: 4.5, loud: 4.8}" in prompt


def test_build_expert_prompt_ab():
    """Verify the A/B testing expert prompt builds correctly with 5 factors per prompt."""
    prompt = build_expert_prompt_ab(
        A_mos=3.5, A_noi=4.0, A_col=3.0, A_dis=4.5, A_loud=3.8,
        B_mos=4.1, B_noi=4.5, B_col=4.0, B_dis=4.8, B_loud=4.2
    )

    # Assert A/B definitions included
    assert "rating from 1 to 5" in prompt
    assert "(4) dis: the discontinuity in the audio" in prompt
    assert "Input: {A_mos: 3.5, A_noi: 4.0, A_col: 3.0, A_dis: 4.5, A_loud: 3.8, B_mos: 4.1, B_noi: 4.5, B_col: 4.0, B_dis: 4.8, B_loud: 4.2}" in prompt
