import json
from pathlib import Path


def generate_targets(
    input_json: str = "data/processed/temporal/temporal_metadata_raw.json",
    output_jsonl: str = "data/processed/sft/train_temporal_alld.json",
):
    """Converts Phase 1 metadata into LLM formatted textual targets for ALLD integration."""
    print(f"Reading {input_json}...")
    with open(input_json, "r") as f:
        metadata = json.load(f)

    print(f"Distilling {len(metadata)} targets...")

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for idx, item in enumerate(metadata):
            audio_path = item["audio_path"]
            # Support both old 'noise_type' and new 'degradation_type'
            degradation_type = str(
                item.get("degradation_type", item.get("noise_type", "noise"))
            ).replace("_", " ")
            start_time = f"{float(item['start_time']):.2f}"
            end_time = f"{float(item['end_time']):.2f}"

            # Formatting the response to be natural for different degradation types
            response_text = f"The overall speech is clear, but the quality is interrupted by {degradation_type} occurring between <|{start_time}|> and <|{end_time}|>."

            # Match SFT format:
            # { "audios": ["path/to"], "response": "..." }
            obj = {
                "id": f"temporal_{idx:05d}",
                "audios": [audio_path],
                "response": response_text,
            }

            f.write(json.dumps(obj) + "\n")

    print(f"Saved distilled dataset to {output_jsonl}")


if __name__ == "__main__":
    import typer

    typer.run(generate_targets)
