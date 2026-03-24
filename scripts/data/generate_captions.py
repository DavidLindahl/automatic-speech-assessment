"""
generate_captions.py - CLI tool to generate captions using Gemini.
"""

import os
import typer
from asa.caption_generator import process_single_file

app = typer.Typer()


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
