import csv
import logging
from pathlib import Path
from typing import Optional

import typer

from asa.inference import load_model, run_inference
from asa.processed_data import (
    load_processed_records,
    resolve_audio_path,
    write_processed_records,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _build_dis_index(nisqa_file_csv: Path) -> dict[str, float]:
    """Map degraded-clip basename to its NISQA discontinuity (`dis`) score.

    Raw NISQA-SIM ships `dis` for every clip in ``NISQA_TRAIN_SIM_file.csv``
    (keyed by ``filename_deg``), but the released caption set drops it. This
    join re-adds it for the discontinuity-deviation ablation. Rows with a
    missing or unparseable `dis` are skipped.
    """
    index: dict[str, float] = {}
    with open(nisqa_file_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            deg = row.get("filename_deg")
            if not deg:
                continue
            try:
                index[Path(deg).name] = float(row["dis"])
            except (KeyError, TypeError, ValueError):
                continue
    return index

app = typer.Typer(
    help="Generate DPO dataset by running inference on the SFT warmup model."
)


@app.command()
def generate(
    input_json: Path = typer.Option(
        Path("data/processed/sft/train_nisqa_llama_10k.json"), help="Input dataset."
    ),
    output_json: Path = typer.Option(
        Path("data/processed/dpo/train_dpo_10k.json"), help="Output DPO dataset."
    ),
    model_path: Path = typer.Option(
        Path("models/sft_warmup"), help="Path to the trained model."
    ),
    data_root: Path = typer.Option(
        Path("data"), help="Root directory that contains the raw audio tree."
    ),
    batch_size: int = typer.Option(8, help="Inference batch size."),
    max_samples: Optional[int] = typer.Option(
        None, help="Max samples to process (for debugging)."
    ),
    do_sample: bool = typer.Option(
        True, help="Enable nucleus sampling for diverse rejecteds."
    ),
    temperature: float = typer.Option(1.1, help="Sampling temperature (paper: 1.1)."),
    top_p: float = typer.Option(0.9, help="Nucleus sampling top_p (paper: 0.9)."),
    add_discontinuity_from: Optional[Path] = typer.Option(
        None,
        help=(
            "Path to NISQA_TRAIN_SIM_file.csv. When set, joins each clip's `dis` "
            "score onto the output records for the 5-dim discontinuity ablation. "
            "Leave unset for the paper-faithful 4-dim dataset."
        ),
    ),
):
    """Run inference to generate 'rejected' responses and create a DPO dataset."""

    output_json.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading input data from {input_json}")
    data = load_processed_records(input_json)

    if max_samples:
        data = data[:max_samples]
        logging.info(f"Limited generation to {max_samples} samples.")

    logging.info(f"Loading model from {model_path}")
    processor, model, device = load_model(model_path)

    audio_paths = [
        str(resolve_audio_path(item["audios"][0], data_root)) for item in data
    ]

    logging.info(
        f"Running inference on {len(audio_paths)} samples with batch size {batch_size}..."
    )
    predictions = run_inference(
        model=model,
        processor=processor,
        audio_paths=audio_paths,
        device=device,
        batch_size=batch_size,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    dis_index: dict[str, float] = {}
    if add_discontinuity_from is not None:
        dis_index = _build_dis_index(add_discontinuity_from)
        logging.info(
            f"Loaded {len(dis_index)} discontinuity (`dis`) scores from "
            f"{add_discontinuity_from} for the 5-dim ablation."
        )

    logging.info("Formatting DPO dataset...")
    dpo_data = []
    missing_dis = 0
    for item, pred in zip(data, predictions):
        dpo_item = item.copy()

        # Standardize for DPO
        dpo_item["chosen"] = item["response"]
        dpo_item["rejected"] = pred.strip()

        # Discontinuity-deviation ablation: join `dis` by degraded basename.
        if dis_index:
            deg = Path(str(item["audios"][0])).name
            if deg in dis_index:
                dpo_item["dis"] = dis_index[deg]
            else:
                missing_dis += 1

        dpo_data.append(dpo_item)

    if dis_index and missing_dis:
        logging.warning(
            f"{missing_dis} records had no matching `dis` score and were left "
            "without it; the 5-dim reference prompt will KeyError on these."
        )

    logging.info(f"Saving DPO dataset to {output_json}")
    write_processed_records(output_json, dpo_data)

    logging.info("Done.")


if __name__ == "__main__":
    app()
