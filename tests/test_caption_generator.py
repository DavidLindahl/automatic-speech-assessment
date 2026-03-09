import json
from src.asa.caption_generator import generate_mos_prediction_prompt, generate_ab_test_prompt

def test_generate_mos_prompt_4_factor():
    """Verify that the generated prompt uses explicitly 4 factors and excludes 'dis'."""
    metadata = {"mos": 4.0, "noi": 4.5, "col": 4.8, "loud": 4.2}
    prompt = generate_mos_prediction_prompt(metadata)
    
    # Assert 4 factors check
    assert "4 factors" in prompt.lower()
    
    # Ensure standard elements exist
    assert "noise" in prompt.lower()
    assert "coloration" in prompt.lower()
    assert "loudness" in prompt.lower()
    
    # Ensure that it strips discontinuity
    assert "discontinuity" not in prompt.lower()
    assert "dis:" not in prompt.lower() 
    assert "(4) dis" not in prompt.lower()
    assert "(5) dis" not in prompt.lower()
    
    # Ensure JSON dump contains the original keys without mutation
    assert '"mos": 4.0' in prompt
    assert '"loud": 4.2' in prompt

def test_generate_ab_test_prompt_4_factor():
    """Verify that A/B prompt excludes 'dis' and uses exactly 4 factors."""
    metadata_a = {"mos": 3.0, "noi": 3.5, "col": 3.8, "loud": 3.2}
    metadata_b = {"mos": 4.0, "noi": 4.5, "col": 4.8, "loud": 4.2}
    prompt = generate_ab_test_prompt(metadata_a, metadata_b)
    
    # Check bounds
    assert "4 factors are rating" in prompt
    
    # Check elements included
    assert "noise, coloration, and loudness" in prompt
    
    # Ensure discontinuity is stripped
    assert "discontinuity" not in prompt.lower()
    assert "dis:" not in prompt.lower()
    
    # Ensure data objects inject correctly
    assert '"noi": 3.5' in prompt
    assert '"noi": 4.5' in prompt
