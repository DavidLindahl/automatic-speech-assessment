from pathlib import Path
import os
import json
import typer
import sys

# Add project root to sys.path to allow running as script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

app = typer.Typer()


def convert_to_hf_dataset(json_path: Path, output_dir: Path, out_name: str):
    """
    Converts a JSON/JSONL file containing MOS or A/B text data into a Hugging Face Dataset.
    Applies the Qwen2-Audio conversational format natively supported by TRL SFTTrainer:
    [
      {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "..."}]},
      {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
    ]
    """
    import datasets

    if not json_path.exists():
        print(f"Dataset JSON not found: {json_path}")
        return

    # 1. Load data
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            items = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            items = [json.loads(line.strip()) for line in f if line.strip()]

    # 2. Transform to HF format
    hf_data = {"messages": [], "audios": []}

    global project_root

    for item in items:
        query_text = item.get(
            "query", "Please describe and evaluate the synthetic speech<audio>."
        )
        # Remove the <audio> tag from the text prompt to avoid processor double-injecting <|audio_bos|>
        query_text_clean = query_text.replace("<audio>", "").replace("  ", " ").strip()

        response_text = item["response"]

        user_content = []
        audio_paths = []

        # Determine local audio paths and add 'audio' type to user prompt
        for audio_path_str in item.get("audios", []):
            if "NISQA_Corpus" in audio_path_str:
                rel_path = audio_path_str[audio_path_str.find("NISQA_Corpus") :]
                local_path = Path(project_root) / "data" / "raw" / rel_path
            else:
                file_name = Path(audio_path_str).name
                local_path = Path(project_root) / "data" / "raw" / file_name

            if not local_path.exists():
                print(
                    f"WARNING: Audio file not found locally: {local_path} (mapped from {audio_path_str})"
                )
                continue

            audio_paths.append(str(local_path))
            user_content.append({"type": "audio"})

        # Append instruction text
        user_content.append({"type": "text", "text": query_text_clean})

        # Build conversational messages array
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": response_text}]},
        ]

        hf_data["messages"].append(messages)
        hf_data["audios"].append(audio_paths)

    # 3. Create Dataset and Cast Features
    ds = datasets.Dataset.from_dict(hf_data)

    # Cast audio path strings natively to Apache Arrow audio binaries with downsampling to 16kHz
    ds = ds.cast_column(
        "audios", datasets.Sequence(datasets.Audio(sampling_rate=16000))
    )

    # 4. Save to Disk
    parquet_out = output_dir / f"{out_name}.parquet"
    print(f"Exporting to {parquet_out} with {len(ds)} rows...")
    ds.to_parquet(parquet_out)


@app.command()
def download(
    bucket_name: str = "nisqa-dataset",
    source_blob_name: str = ".",
    destination_path: Path = Path("data/raw"),
):
    """Downloads data from Google Cloud Storage to a local directory."""
    from google.cloud import storage

    print(
        f"Downloading from gs://{bucket_name}/{source_blob_name} to {destination_path}..."
    )

    # Ensure destination exists
    destination_path.mkdir(parents=True, exist_ok=True)

    try:
        client = storage.Client()
    except Exception:
        print("No credentials found. Using anonymous access...")
        client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(
        prefix=source_blob_name if source_blob_name != "." else None
    )
    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        relative_path = os.path.relpath(blob.name, source_blob_name)
        local_path = destination_path / relative_path

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {blob.name} to {local_path}...")
        blob.download_to_filename(str(local_path))


@app.command()
def preprocess_sft(
    data_path: Path = typer.Argument(Path("data/raw")),
    output_folder: Path = typer.Argument(Path("data/processed")),
) -> None:
    """Stage 1: Preprocesses dataset for Supervised Fine-Tuning (SFT)."""
    print("Preprocessing data for SFT...")

    # 1. Run Sampler
    print("\n--- Step 1: Sampling Data ---")
    from src.asa.sampler import sample_data

    nisqa_corpus_path = data_path / "NISQA_Corpus"

    sample_data(nisqa_corpus_path, output_folder)

    # 2. Run Caption Generator
    print("\n--- Step 2: Generating Captions ---")
    from src.asa.caption_generator import process_single_file

    # Process MOS Dataset
    mos_input = output_folder / "mos_dataset.json"
    mos_output = output_folder / "train_nisqa_llama_10k.json"
    mos_parquet = output_folder / "train_nisqa_llama_10k.parquet"
    if mos_input.exists():
        if not mos_output.exists():
            print(f"Generating captions for MOS dataset: {mos_input} -> {mos_output}")
            process_single_file(str(mos_input), str(mos_output))
        else:
            print(
                f"Captions for MOS dataset already exist at {mos_output}, skipping generation."
            )

    # Process A/B Dataset
    ab_input = output_folder / "ab_dataset.json"
    ab_output = output_folder / "train_nisqa_abtest_llama_10k.json"
    ab_parquet = output_folder / "train_nisqa_abtest_llama_10k.parquet"
    if ab_input.exists():
        if not ab_output.exists():
            print(f"Generating captions for A/B dataset: {ab_input} -> {ab_output}")
            process_single_file(str(ab_input), str(ab_output))
        else:
            print(
                f"Captions for A/B dataset already exist at {ab_output}, skipping generation."
            )

    # 3. Convert to HF Parquet
    print("\n--- Step 3: Converting to HF Parquet Datasets ---")

    if mos_input.exists():
        if not mos_parquet.exists():
            print("Converting MOS JSON to HF Parquet Dataset...")
            convert_to_hf_dataset(mos_output, output_folder, "train_nisqa_llama_10k")
        else:
            print(
                f"HF Parquet Dataset for MOS already exists at {mos_parquet}, skipping conversion."
            )

    if ab_input.exists():
        if not ab_parquet.exists():
            print("Converting A/B JSON to HF Parquet Dataset...")
            convert_to_hf_dataset(
                ab_output, output_folder, "train_nisqa_abtest_llama_10k"
            )
        else:
            print(
                f"HF Parquet Dataset for A/B already exists at {ab_parquet}, skipping conversion."
            )

    print("\nSFT Preprocessing pipeline complete.")


@app.command()
def preprocess_alld(
    data_path: Path = typer.Argument(Path("data/processed")),
    output_folder: Path = typer.Argument(Path("data/processed")),
) -> None:
    """Stage 2: Preprocesses dataset for ALLD (RLHF/DPO) pipeline."""
    print("Generating preference data for ALLD (DPO)...")
    # TODO: Implement step 2 data generation here:
    # 1. Load SFT warmup checkpoint
    # 2. Iterate over training meta information
    # 3. Generate preferred/dispreferred completions (y_t and y_a) using expert LLM
    # 4. Save to `data/processed/dpo_dataset.parquet`
    print("\nALLD Preprocessing pipeline complete.")


if __name__ == "__main__":
    app()
