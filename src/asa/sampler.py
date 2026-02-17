import typer
import pandas as pd
from pathlib import Path
import json
import random

app = typer.Typer()


def load_csv(
    data_root: Path, corpus: str, file_pattern: str = "*_file.csv"
) -> pd.DataFrame:
    """Load the file CSV for a given corpus."""
    csv_path = list((data_root / corpus).rglob(file_pattern))[0]
    return pd.read_csv(csv_path)


@app.command()
def sample_data(
    data_root: Path = Path("data/raw/NISQA_Corpus"),
    output_dir: Path = Path("data/processed"),
    seed: int = 42,
):
    """
    Sample data from NISQA corpus for LLM training/testing.
    Generates A/B test pairs (with MOS gap > 0.5) and MOS prediction samples.
    Saves to separate JSON files.
    """
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load NISQA_TRAIN_SIM
    print("Loading NISQA_TRAIN_SIM...")
    df_sim = load_csv(data_root, "NISQA_TRAIN_SIM")

    # Shuffle
    df_sim = df_sim.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Define sizes
    n_ab_pairs = 500
    n_mos_train = 500
    n_mos_test_in = 250
    n_mos_test_out = 250

    # --- A/B Test Set (500 pairs) ---
    print(f"Sampling {n_ab_pairs} A/B pairs with MOS gap > 0.5...")
    ab_test_data = []
    used_indices = set()

    # We iterate through the shuffled dataframe to find pairs
    # This is a simple greedy approach.
    # To avoid reusing samples too much, we'll try to use unique samples if possible,
    # but for 500 pairs (1000 samples) from ~10k, it's easy.

    # Let's take a chunk of data to pair up
    # We need 1000 items ideally.
    # We will iterate through indices i and i+1. If gap is good, take it.
    # If not, try i and i+2, etc.

    idx = 0
    while len(ab_test_data) < n_ab_pairs and idx < len(df_sim) - 1:
        if idx in used_indices:
            idx += 1
            continue

        row_a = df_sim.iloc[idx]

        # Look ahead for a match
        for offset in range(1, 100):  # limited lookahead to keep it fast
            curr_b_idx = idx + offset
            if curr_b_idx >= len(df_sim):
                break

            if curr_b_idx in used_indices:
                continue

            row_b = df_sim.iloc[curr_b_idx]

            if abs(row_a["mos"] - row_b["mos"]) > 0.5:
                # Found a pair
                used_indices.add(idx)
                used_indices.add(curr_b_idx)

                meta_a = {
                    "mos": row_a["mos"],
                    "noi": row_a["noi"],
                    "col": row_a["col"],
                    "dis": row_a["dis"],
                    "loud": row_a["loud"],
                }
                meta_b = {
                    "mos": row_b["mos"],
                    "noi": row_b["noi"],
                    "col": row_b["col"],
                    "dis": row_b["dis"],
                    "loud": row_b["loud"],
                }

                # Determine winner
                if row_a["mos"] > row_b["mos"]:
                    winner = "A"
                else:
                    winner = "B"

                pair_data = {
                    "pair_id": f"ab_{len(ab_test_data)}",
                    "audio_a_path": str(data_root / row_a["filepath_deg"]),
                    "audio_b_path": str(data_root / row_b["filepath_deg"]),
                    "meta_a": meta_a,
                    "meta_b": meta_b,
                    "winner": winner,
                    "split": "train_sim",
                }
                ab_test_data.append(pair_data)
                break

        idx += 1  # Move to next primary candidate

    if len(ab_test_data) < n_ab_pairs:
        print(f"Warning: Only found {len(ab_test_data)} pairs with suitable gap.")

    # --- MOS Data (Disjoint) ---
    print("Sampling MOS data (disjoint from A/B)...")

    # Filter out used indices
    available_indices = [i for i in range(len(df_sim)) if i not in used_indices]
    df_remaining = df_sim.iloc[available_indices]

    if len(df_remaining) < (n_mos_train + n_mos_test_in):
        raise ValueError("Not enough data remaining for MOS sets after A/B sampling.")

    df_mos_train = df_remaining.iloc[:n_mos_train]
    df_mos_test_in = df_remaining.iloc[n_mos_train : n_mos_train + n_mos_test_in]

    mos_data = []

    def add_to_mos_data(row, split):
        mos_data.append(
            {
                "utt_id": row["filename_deg"],
                "audio_path": str(data_root / row["filepath_deg"]),
                "meta": {
                    "mos": row["mos"],
                    "noi": row["noi"],
                    "col": row["col"],
                    "dis": row["dis"],
                    "loud": row["loud"],
                },
                "split": split,
            }
        )

    print(f"Processing {len(df_mos_train)} train_sim samples...")
    df_mos_train.apply(lambda row: add_to_mos_data(row, "train_sim"), axis=1)

    print(f"Processing {len(df_mos_test_in)} test_in_domain samples...")
    df_mos_test_in.apply(lambda row: add_to_mos_data(row, "test_in_domain"), axis=1)

    # --- MOS Test Out-of-Domain (250 samples total) ---
    out_datasets = [
        ("NISQA_TRAIN_LIVE", "val_live"),
        ("NISQA_TEST_P501", "test_p501"),
        ("NISQA_TEST_FOR", "test_for"),
    ]

    samples_per_ds = n_mos_test_out // len(out_datasets)  # 83
    remainder = n_mos_test_out % len(out_datasets)  # 1

    for idx, (corpus, split_name) in enumerate(out_datasets):
        print(f"Loading {corpus}...")
        df_out = load_csv(data_root, corpus)

        n_sample = samples_per_ds + (1 if idx < remainder else 0)

        # Sample
        df_sampled = df_out.sample(n=n_sample, random_state=seed)

        print(f"Processing {len(df_sampled)} samples for {split_name}...")
        df_sampled.apply(lambda row: add_to_mos_data(row, split_name), axis=1)

    # Save to files
    ab_file = output_dir / "ab_dataset.json"
    mos_file = output_dir / "mos_dataset.json"

    with open(ab_file, "w") as f:
        json.dump(ab_test_data, f, indent=2)

    with open(mos_file, "w") as f:
        json.dump(mos_data, f, indent=2)

    print(f"\nSaved datasets to {output_dir}")
    print(f"  {ab_file}: {len(ab_test_data)} pairs")
    print(f"  {mos_file}: {len(mos_data)} samples")
    print("    Splits:")
    split_counts = pd.DataFrame(mos_data)["split"].value_counts()
    print(split_counts)


if __name__ == "__main__":
    app()
