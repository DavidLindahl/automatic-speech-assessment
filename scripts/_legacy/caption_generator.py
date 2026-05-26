#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speech Quality Descriptive Caption Generator
This script generates descriptive captions for speech quality evaluation using Google Gemini models.
It supports both individual speech quality evaluation (MOS prediction) and A/B testing between two speech samples.
"""

import os
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import typer
import pandas as pd
from google import genai
from google.genai import types

# Legacy NISQA preprocessing tool, moved out of src/asa/ as it's no longer
# on the active path. Self-bootstraps by adding repo/src to sys.path so it
# can run standalone via `python scripts/_legacy/caption_generator.py ...`.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from asa.processed_data import write_processed_records

# Configure Gemini API
# Assumes GEMINI_API_KEY is set in environment variables
_GEMINI_CLIENT: Optional[genai.Client] = None


def get_client() -> genai.Client:
    """Return a singleton Gemini client configured from env."""
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "Warning: GEMINI_API_KEY environment variable not found. API calls may fail."
        )
    _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


# Model configuration
MODEL_NAME = "gemini-3.1-flash-lite-preview"

app = typer.Typer()


# Temp and TopP is based on paper's second itteration
def call_gemini_api(prompt: str, temperature: float = 1.1, top_p: float = 0.90) -> str:
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
        client = get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
            ),
        )
        return response.text or ""
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
        metadata: A dictionary containing 'mos', 'noi', 'col', and 'loud' values
        example_data: Optional example data point to include in the prompt
        example_response: Optional example response to include in the prompt

    Returns:
        The formatted prompt string
    """
    prompt = """I will give you a tuple of meta information for speech quality evaluation, it contains 4 factors are rating from 1 to 5. For all these factors, higher is better.
(1) mos: the overall quality. 1 is very bad, 2 is poor, 3 is fair, 4 is good, 5 is excellent.
(2) noi: the level of noise in the audio, reflecting the impact of background noise or other non-speech interference on audio quality. 1 is very noisy, 2 is somewhat noisy, 3 is neither noisy nor clean, 4 is somewhat clean, and 5 is completely clean.
(3) col: the alterations in the natural sound of speech caused by distortions or unwanted modifications. 1 is severely distorted, 2 is significantly distorted, 3 is moderately distorted, 4 is slightly distorted, and 5 is no distortion.
(4) loud: the perceived volume or loudness of the audio. 1 is extremely quiet, 2 is significantly quiet, 3 is soft but understandable, 4 is clearly loud, and 5 is perfectly loud.
I need you to generate a descriptive evaluation for this speech, including a description according to the score from noise, coloration, and loudness, analyze how they influence the overall quality, and add the mos in the end."""

    default_example = {"mos": 1.8, "noi": 3.2, "col": 1.9, "loud": 3.3}
    default_response = "This synthesized speech has a moderate level of noise and volume. However, it suffers from significant distortion, which greatly impacts its overall quality. With an overall MOS score of 1.8, it falls into the somewhat unnatural category."

    # Add example if provided, otherwise use default
    data_to_use = example_data if example_data else default_example
    response_to_use = example_response if example_response else default_response

    prompt += f"\nFor example, input is {json.dumps(data_to_use)}, then you should output: {response_to_use}"

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
        metadata_a: A dictionary containing 'mos', 'noi', 'col', and 'loud' values for Speech A
        metadata_b: A dictionary containing 'mos', 'noi', 'col', and 'loud' values for Speech B

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

    # Normalize result to "A" or "B"
    clean_result = result.strip()
    if "[SpeechA]" in clean_result or "SpeechA" in clean_result:
        return "A"
    elif "[SpeechB]" in clean_result or "SpeechB" in clean_result:
        return "B"

    # Fallback if model output is unexpected, though prompt instruction is strict
    return clean_result


def process_single_file(input_path: str, output_path: str):
    """
    Process a single dataset JSON file, generate captions/evaluations using Gemini, and save results.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    with open(input_path, "r") as f:
        data = json.load(f)

    # If the input is a list of JSON objects (which is expected), process each item
    if not isinstance(data, list):
        print(
            "Warning: Input JSON is not a list. Attempting to process as single item if valid."
        )
        data = [data]

    results = []

    print(f"Processing {len(data)} items from {input_path}...")

    for i, item in enumerate(data):
        # Create a new result dict rather than modifying inplace, to match target schema
        result_item = {}

        # Check if it's a single utterance (MOS Prediction)
        if "meta" in item:
            identifier = item.get("utt_id", f"item_{i+1}")
            print(f"Processing {identifier} (MOS Prediction)...")
            metadata = item["meta"]

            prompt = generate_mos_prediction_prompt(metadata)
            response = call_gemini_api(prompt)
            # DEBUG:
            print(response)
            if "audio_path" in item:
                result_item["audios"] = [item["audio_path"]]
            elif "audios" in item:
                result_item["audios"] = item["audios"]
            else:
                result_item["audios"] = []

            result_item["response"] = response
            result_item["query"] = (
                "Please describe and evaluate the synthetic speech<audio>."
            )

            # Flatten metadata
            for k, v in metadata.items():
                result_item[k] = v

            # Copy other possible fields (like utt_id, spread info etc if needed)
            if "utt_id" in item:
                result_item["utt_id"] = item["utt_id"]
            if "split" in item:
                result_item["split"] = item["split"]

        # Check if it's an A/B pair (A/B Test)
        elif "meta_a" in item and "meta_b" in item:
            identifier = item.get("pair_id", f"pair_{i+1}")
            print(f"Processing {identifier} (A/B Test)...")
            metadata_a = item["meta_a"]
            metadata_b = item["meta_b"]

            ab_prompt = generate_ab_test_prompt(metadata_a, metadata_b)
            ab_result = call_gemini_api(ab_prompt)
            ab_summary = summarize_ab_test(ab_result)

            # Construct output item matching train_nisqa_abtest_llama_10k.json structure
            audios = []
            if "audio_a_path" in item:
                audios.append(item["audio_a_path"])
            if "audio_b_path" in item:
                audios.append(item["audio_b_path"])
            result_item["audios"] = audios

            result_item["response"] = ab_result
            result_item["query"] = (
                "Please perform A/B preference test between<audio>and<audio>, including a tie."
            )

            # Flatten metadata with A_ and B_ prefixes
            for k, v in metadata_a.items():
                result_item[f"A_{k}"] = v
            for k, v in metadata_b.items():
                result_item[f"B_{k}"] = v

            # Copy winner if ground truth exists (it's in input)
            if "winner" in item:
                result_item["winner"] = item["winner"]

            # Store prediction separately? Or replace response?
            # The target file has "winner": "B", which is likely ground truth.
            # We generated a response. The response usually indicates prediction.
            # We can store our predicted winner separately if needed.
            result_item["winner_predicted"] = ab_summary

            if "pair_id" in item:
                result_item["pair_id"] = item["pair_id"]
            if "split" in item:
                result_item["split"] = item["split"]

        else:
            print(f"Skipping item {i+1}: Unknown format.")
            continue

        results.append(result_item)
        print(f"Processed {i+1}/{len(data)} items.")

    # Save results
    write_processed_records(output_path, results)

    print(f"Processing complete. Results saved to {output_path}")


META_FIELDS = ("mos", "noi", "col", "dis", "loud")


def row_to_meta(row: pd.Series) -> Dict[str, float]:
    """Convert a CSV row into a metadata dict."""
    return {field: float(row[field]) for field in META_FIELDS}


def sample_ab_pairs_for_corpus(
    corpus: str,
    data_root: Path,
    pairs_per_corpus: Optional[int],
    min_mos_gap: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample non-overlapping A/B pairs from one NISQA corpus with MOS-gap filtering."""
    csv_path = data_root / corpus / f"{corpus}_file.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    df = pd.read_csv(csv_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    used_indices = set()
    pairs: List[Dict[str, Any]] = []

    for idx in range(len(df)):
        if idx in used_indices:
            continue
        if pairs_per_corpus is not None and len(pairs) >= pairs_per_corpus:
            break

        row_a = df.iloc[idx]
        for idx_b in range(idx + 1, len(df)):
            if idx_b in used_indices:
                continue
            row_b = df.iloc[idx_b]
            if abs(float(row_a["mos"]) - float(row_b["mos"])) < min_mos_gap:
                continue

            used_indices.update((idx, idx_b))
            if float(row_a["mos"]) > float(row_b["mos"]):
                winner = "A"
            elif float(row_b["mos"]) > float(row_a["mos"]):
                winner = "B"
            else:
                winner = "TIE"

            pairs.append(
                {
                    "audio_a_path": str(data_root / corpus / row_a["filepath_deg"]),
                    "audio_b_path": str(data_root / corpus / row_b["filepath_deg"]),
                    "meta_a": row_to_meta(row_a),
                    "meta_b": row_to_meta(row_b),
                    "winner": winner,
                }
            )
            break

    return pairs


DEFAULT_AB_TEST_CORPORA = ["NISQA_TEST_P501", "NISQA_VAL_LIVE", "NISQA_TEST_FOR"]


@app.command()
def generate_ab_test_set(
    data_root: Path = typer.Option(
        Path("data/raw/NISQA_Corpus"),
        help="Root folder containing NISQA corpus subfolders.",
    ),
    corpora: str = typer.Option(
        ",".join(DEFAULT_AB_TEST_CORPORA),
        help="Comma-separated corpus names to sample A/B pairs from.",
    ),
    min_mos_gap: float = typer.Option(
        0.5, help="Minimum MOS gap required between SpeechA and SpeechB."
    ),
    seed: int = typer.Option(42, help="Random seed for pair sampling."),
    pairs_output: Path = typer.Option(
        Path("data/processed/ab-test-set-pairs.json"),
        help="Path to save sampled raw A/B pairs.",
    ),
    captions_output: Path = typer.Option(
        Path("data/processed/intermediate/ab-test-set-captions.json"),
        help="Path to save generated A/B caption records (JSONL).",
    ),
    max_samples: Optional[int] = typer.Option(
        None,
        help="Cap number of pairs to caption. Useful for test runs (e.g. --max-samples 5).",
    ),
) -> None:
    """Sample out-of-domain AB pairs and generate captions."""
    selected_corpora = [c.strip() for c in corpora.split(",") if c.strip()]
    if not selected_corpora:
        raise typer.BadParameter("No corpora provided.")

    pair_limit = None
    all_pairs: List[Dict[str, Any]] = []

    for i, corpus in enumerate(selected_corpora):
        corpus_pairs = sample_ab_pairs_for_corpus(
            corpus=corpus,
            data_root=data_root,
            pairs_per_corpus=pair_limit,
            min_mos_gap=min_mos_gap,
            seed=seed + i,
        )
        all_pairs.extend(corpus_pairs)
        print(f"Sampled {len(corpus_pairs)} A/B pairs from {corpus}")

    pairs_output.parent.mkdir(parents=True, exist_ok=True)
    pairs_output.write_text(json.dumps(all_pairs, indent=2), encoding="utf-8")
    print(f"Saved {len(all_pairs)} raw pairs to {pairs_output}")

    pairs_to_caption = all_pairs if max_samples is None else all_pairs[:max_samples]
    print(
        f"Generating Gemini captions for {len(pairs_to_caption)}/{len(all_pairs)} A/B pairs..."
    )
    results = []
    for i, item in enumerate(pairs_to_caption, start=1):
        ab_prompt = generate_ab_test_prompt(item["meta_a"], item["meta_b"])
        ab_result = call_gemini_api(ab_prompt)
        ab_summary = summarize_ab_test(ab_result)

        result_item: Dict[str, Any] = {
            "audios": [item["audio_a_path"], item["audio_b_path"]],
            "response": ab_result,
            "query": "Please perform A/B preference test between<audio>and<audio>, including a tie.",
            "winner": item["winner"],
            "winner_predicted": ab_summary,
        }
        result_item.update({f"A_{k}": v for k, v in item["meta_a"].items()})
        result_item.update({f"B_{k}": v for k, v in item["meta_b"].items()})
        results.append(result_item)
        print(f"Processed {i}/{len(pairs_to_caption)}")

    write_processed_records(captions_output, results)
    print(f"Saved A/B caption dataset to {captions_output}")


@app.command()
def process_data(
    data_dir: str = typer.Option(
        "data/processed",
        "--data-dir",
        "-d",
        help="Directory containing input JSON files (mos_dataset.json, ab_dataset.json).",
    ),
):
    """
    Process dataset JSONs (mos_dataset.json, ab_dataset.json) in the specified directory,
    generate captions/evaluations using Gemini, and save to target files:
    - train_nisqa_llama_10k.json
    - train_nisqa_abtest_llama_10k.json
    """
    data_path = os.path.abspath(data_dir)
    # 1. Process MOS Dataset
    mos_input = os.path.join(data_path, "mos_dataset.json")
    mos_output = os.path.join(data_path, "train_nisqa_llama_10k.json")

    if os.path.exists(mos_input):
        print(f"Found {mos_input}. Processing to {mos_output}...")
        process_single_file(mos_input, mos_output)
    else:
        print(f"Skipping MOS dataset: {mos_input} not found.")

    # 2. Process A/B Dataset
    ab_input = os.path.join(data_path, "ab_dataset.json")
    ab_output = os.path.join(data_path, "train_nisqa_abtest_llama_10k.json")

    if os.path.exists(ab_input):
        print(f"Found {ab_input}. Processing to {ab_output}...")
        process_single_file(ab_input, ab_output)
    else:
        print(f"Skipping A/B dataset: {ab_input} not found.")


if __name__ == "__main__":
    app()
