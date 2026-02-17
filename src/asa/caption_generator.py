#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speech Quality Descriptive Caption Generator
This script generates descriptive captions for speech quality evaluation using Google Gemini models.
It supports both individual speech quality evaluation (MOS prediction) and A/B testing between two speech samples.
"""

import os
import argparse
import json
from typing import Dict, Optional
import google.generativeai as genai

# Configure Gemini API
# Assumes GEMINI_API_KEY is set in environment variables
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
else:
    print("Warning: GEMINI_API_KEY environment variable not found. API calls may fail.")

# Model configuration
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"


def call_gemini_api(prompt: str, temperature: float = 1.0, top_p: float = 0.95) -> str:
    """
    Call the Google Gemini model with the given prompt.

    Args:
        prompt: The input prompt for the model
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter

    Returns:
        The model's response as a string
    """
    try:
        model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
            ),
        )
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"Error: {str(e)}"


def generate_mos_prediction_prompt(
    metadata: Dict[str, float],
    example_data: Optional[Dict] = None,
    example_response: Optional[str] = None,
) -> str:
    """
    Generate a prompt for MOS prediction based on the metadata.

    Args:
        metadata: A dictionary containing 'mos', 'noi', 'col', 'dis', 'loud' values
        example_data: Optional example data point to include in the prompt
        example_response: Optional example response to include in the prompt

    Returns:
        The formatted prompt string
    """
    prompt = """I will give you a tuple of meta information for speech quality evaluation, it contains 5 factors are rating from 1 to 5. For all these factors, higher is better.
(1) mos: the overall quality. 1 is very bad, 2 is poor, 3 is fair, 4 is good, 5 is excellent.
(2) noi: the level of noise in the audio, reflecting the impact of background noise or other non-speech interference on audio quality. 1 is very noisy, 2 is somewhat noisy, 3 is neither noisy nor clean, 4 is somewhat clean, and 5 is completely clean.
(3) col: the alterations in the natural sound of speech caused by distortions or unwanted modifications. 1 is severely distorted, 2 is significantly distorted, 3 is moderately distorted, 4 is slightly distorted, and 5 is no distortion.
(4) dis: the discontinuity in the audio, reflecting whether there are breaks, stutters, or incoherence during playback. 1 is severely discontinuous, 2 is significantly discontinuous, 3 is moderately discontinuous, 4 is slightly discontinuous, and 5 is no discontinuity.
(5) loud: the perceived volume or loudness of the audio. 1 is extremely quiet, 2 is significantly quiet, 3 is soft but understandable, 4 is clearly loud, and 5 is perfectly loud.
I need you to generate a descriptive evaluation for this speech, including a description according to the score from (2) to (5), analyze how they influence the overall quality, and add the mos in the end."""

    # Add example if provided
    if example_data and example_response:
        prompt += f"\nFor example, input is {json.dumps(example_data)}, then you should output: {example_response}"

    # Add current data point
    prompt += (
        f"\nNow the input is {json.dumps(metadata)}. Please only output the evaluation:"
    )

    return prompt


def generate_ab_test_prompt(
    metadata_a: Dict[str, float], metadata_b: Dict[str, float]
) -> str:
    """
    Generate a prompt for A/B testing based on the metadata of two speech samples.

    Args:
        metadata_a: A dictionary containing 'mos', 'noi', 'col', 'dis', 'loud' values for Speech A
        metadata_b: A dictionary containing 'mos', 'noi', 'col', 'dis', 'loud' values for Speech B

    Returns:
        The formatted prompt string
    """
    prompt = """I will give you a tuple of meta information for speech quality evaluation, it contains 5 factors are rating from 1 to 5. For all these factors, higher is better.
(1) mos: the overall quality. 1 is very bad, 2 is poor, 3 is fair, 4 is good, 5 is excellent.
(2) noi: the level of noise in the audio, reflecting the impact of background noise or other non-speech interference on audio quality. 1 is very noisy, 2 is somewhat noisy, 3 is neither noisy nor clean, 4 is somewhat clean, and 5 is completely clean.
(3) col: the alterations in the natural sound of speech caused by distortions or unwanted modifications. 1 is severely distorted, 2 is significantly distorted, 3 is moderately distorted, 4 is slightly distorted, and 5 is no distortion.
(4) dis: the discontinuity in the audio, reflecting whether there are breaks, stutters, or incoherence during playback. 1 is severely discontinuous, 2 is significantly discontinuous, 3 is moderately discontinuous, 4 is slightly discontinuous, and 5 is no discontinuity.
(5) loud: the perceived volume or loudness of the audio. 1 is extremely quiet, 2 is significantly quiet, 3 is soft but understandable, 4 is clearly loud, and 5 is perfectly loud.
I need you to perform A/B test according to their mos (mos higher means winner). You can flexibly select 1~3 aspects from (2)~(5) with an obvious gap (usually score difference more than 0.5), then compare them according to these distinctions. Finally, please give your preference with a reasonable analysis."""

    # Add metadata for both speech samples
    prompt += f"\nSpeechA: {json.dumps(metadata_a)}"
    prompt += f"\nSpeechB: {json.dumps(metadata_b)}"
    prompt += "\nPlease provide your comparison and determine which speech is better:"

    return prompt


def summarize_ab_test(llm_output: str) -> str:
    """
    Summarize the A/B test result using Gemini.

    Args:
        llm_output: The output from the A/B test generation

    Returns:
        A string with either "[SpeechA]" or "[SpeechB]"
    """
    prompt = f"""According to the context, please judge if SpeechA is better or SpeechB is better. Only output '[SpeechA]' or '[SpeechB]', do not give any analysis.
Context:
{llm_output}"""

    result = call_gemini_api(prompt, temperature=0.7, top_p=0.95)
    return result.strip()


def process_dataset_json(input_json_path: str, output_json_path: str):
    """
    Process dataset JSON, generate captions/evaluations, and save results.

    Args:
        input_json_path: Path to the input JSON file.
        output_json_path: Path to the output JSON file.
    """
    if not os.path.exists(input_json_path):
        print(f"Error: Input file '{input_json_path}' not found.")
        return

    with open(input_json_path, "r") as f:
        data = json.load(f)

    # If the input is a list of JSON objects (which is expected), process each item
    if not isinstance(data, list):
        # Handle case where it might be a single object or different structure
        # adhering to user example which implies a list of objects like the ones shown
        print(
            "Warning: Input JSON is not a list. Attempting to process as single item if valid."
        )
        data = [data]

    results = []

    print(f"Processing {len(data)} items from {input_json_path}...")

    for i, item in enumerate(data):
        result_item = item.copy()  # Preserve original data

        # Check if it's a single utterance
        if "meta" in item:
            # MOS Prediction / Caption Generation
            print(f"Processing item {i+1} (MOS Prediction)...")
            metadata = item["meta"]
            prompt = generate_mos_prediction_prompt(metadata)
            response = call_gemini_api(prompt)

            # Optional: Generate diverse response
            # response_diverse = call_gemini_api(prompt, temperature=1.1, top_p=0.95)

            result_item["generated_caption"] = response
            # result_item["generated_caption_diverse"] = response_diverse

        # Check if it's an A/B pair
        elif "meta_a" in item and "meta_b" in item:
            # A/B Test
            print(f"Processing item {i+1} (A/B Test)...")
            metadata_a = item["meta_a"]
            metadata_b = item["meta_b"]

            ab_prompt = generate_ab_test_prompt(metadata_a, metadata_b)
            ab_result = call_gemini_api(ab_prompt)
            ab_summary = summarize_ab_test(ab_result)

            result_item["ab_evaluation"] = ab_result
            result_item["ab_winner_predicted"] = ab_summary

        else:
            print(f"Skipping item {i+1}: Unknown format.")

        results.append(result_item)

    # Save results
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Processing complete. Results saved to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech quality captions from JSON metadata using Google Gemini models"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/processed/llm_dataset.json",
        help="Path to the input JSON file (default: data/processed/llm_dataset.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/processed/output.json",
        help="Path to the output JSON file (default: data/processed/output.json)",
    )

    args = parser.parse_args()

    process_dataset_json(args.input, args.output)


if __name__ == "__main__":
    main()
