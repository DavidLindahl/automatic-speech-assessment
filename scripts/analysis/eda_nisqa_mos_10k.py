"""Descriptive analysis of the global MOS SFT training set (train_nisqa_llama_10k).

This is the dataset every downstream model is built from: the Qwen2-Audio MOS
captioning SFT set. Each record pairs one NISQA-SIM degraded clip with a query and
a target caption that ends in an explicit MOS score. We characterise the set along
the four axes that decide how the supervision behaves, plus a corpus-composition
panel:

1. MOS distribution           - what quality range the model is trained on.
2. Degradation taxonomy       - the 13 raw NISQA tags and the 6-category collapse
                                used for legibility (the honesty check).
3. Degradations per clip      - how many simultaneous degradations each clip carries.
4. Clip duration              - the absolute time scale of one training example.
5. Corpus source composition  - which speech corpora the clips come from, and how
                                MOS varies across them.

The MOS values and degradation tags are joined 1:1 from the NISQA per-file CSV
(``NISQA_TRAIN_SIM_file.csv``) via the degraded filename. Nothing in this set is
filtered by MOS: the full 1.0-5.0 range is present (the MOS<=3.0 filter is applied
only later, when building the separate temporal-mix subset).

Run locally, no HPC:

    python scripts/analysis/eda_nisqa_mos_10k.py \
        --figures-dir ../_papers/asa-thesis/figures

Figures are written as both PDF (vector, for LaTeX) and PNG. A companion
``eda_nisqa_mos_10k_stats.json`` with every number behind the figures is written
next to the figures so captions can be filled from real values.
"""

from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# The 13 raw NISQA simulation degradation columns, in a readable order grouped by
# the family they collapse into.
DEGRADATION_COLUMNS: list[str] = [
    "bgn",
    "wbgn",
    "p50mnru",
    "codec1",
    "codec2",
    "codec3",
    "plcMode1",
    "plcMode2",
    "plcMode3",
    "filter",
    "arb_filter",
    "timeclipping",
    "clipping",
]

# 13 -> 6 readable category map (matches DEGRADATION_LABELS in the data pipeline).
CATEGORY_MAP: dict[str, str] = {
    "bgn": "background noise",
    "wbgn": "background noise",
    "p50mnru": "background noise",
    "codec1": "codec artifacts",
    "codec2": "codec artifacts",
    "codec3": "codec artifacts",
    "plcMode1": "packet-loss concealment",
    "plcMode2": "packet-loss concealment",
    "plcMode3": "packet-loss concealment",
    "filter": "band-limiting filter",
    "arb_filter": "band-limiting filter",
    "timeclipping": "time clipping",
    "clipping": "clipping distortion",
}

# Stable display order for the six collapsed categories (largest families first).
CATEGORY_ORDER: list[str] = [
    "background noise",
    "codec artifacts",
    "packet-loss concealment",
    "band-limiting filter",
    "clipping distortion",
    "time clipping",
]

# One colour per collapsed category; raw tags inherit their family colour so the
# two panels of the taxonomy figure read as the same grouping.
CATEGORY_COLORS: dict[str, str] = {
    "background noise": "#2f6db5",
    "codec artifacts": "#e07b39",
    "packet-loss concealment": "#5aa469",
    "band-limiting filter": "#9b59b6",
    "clipping distortion": "#c0392b",
    "time clipping": "#7f8c8d",
}

ACCENT = "#2f6db5"
GRID_GREY = "#cccccc"


def set_paper_style() -> None:
    """Apply a clean, publication-oriented matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.color": GRID_GREY,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.6,
        }
    )


def is_active_tag(value: object) -> bool:
    """Return whether a NISQA metadata cell encodes an active degradation."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    token = str(value).strip()
    return token not in {"", "-", "nan", "None", "NaN"}


def load_records(jsonl_path: Path) -> list[dict]:
    """Load the SFT JSONL records (one JSON object per line)."""
    records: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataframe(records: list[dict], nisqa_csv: Path) -> pd.DataFrame:
    """Join SFT records to the NISQA per-file CSV by degraded filename.

    Args:
        records: SFT JSONL records with ``audios`` and ``mos`` fields.
        nisqa_csv: Path to ``NISQA_TRAIN_SIM_file.csv``.

    Returns:
        One row per SFT record with MOS, source corpus, active degradation tags,
        and the collapsed category set.
    """
    nisqa = pd.read_csv(nisqa_csv)
    by_deg = {row["filename_deg"]: row for _, row in nisqa.iterrows()}

    rows: list[dict] = []
    unmatched = 0
    for record in records:
        deg_name = Path(str(record["audios"][0])).name
        meta = by_deg.get(deg_name)
        if meta is None:
            unmatched += 1
            continue
        active = [col for col in DEGRADATION_COLUMNS if is_active_tag(meta.get(col))]
        categories = sorted({CATEGORY_MAP[col] for col in active})
        rows.append(
            {
                "id": record.get("id"),
                "deg_name": deg_name,
                "stored_audio": str(record["audios"][0]),
                "filepath_deg": str(meta.get("filepath_deg", "")),
                "mos": float(record["mos"]),
                "source": str(meta.get("source", "")).strip(),
                "active_tags": active,
                "num_tags": len(active),
                "categories": categories,
                "num_categories": len(categories),
            }
        )

    if unmatched:
        print(f"WARNING: {unmatched} records did not join to the NISQA CSV.")
    return pd.DataFrame(rows)


def wav_duration_seconds(path: Path) -> float:
    """Read a WAV clip's duration from its RIFF header, with no audio library.

    The NISQA clips are 32-bit IEEE-float WAVs (format code 3), which the stdlib
    ``wave`` module refuses to open, so we parse the chunk structure directly:
    locate the ``fmt `` chunk for channels/rate/bit-depth and the ``data`` chunk
    for the sample-byte count, then divide.

    Args:
        path: Path to a RIFF/WAVE file.

    Returns:
        Duration in seconds.

    Raises:
        ValueError: If the file is not a RIFF/WAVE or required chunks are absent.
    """
    with path.open("rb") as handle:
        header = handle.read(12)
        if header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError("not a RIFF/WAVE file")
        channels = sample_rate = bits_per_sample = None
        data_bytes = None
        while True:
            chunk_header = handle.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id = chunk_header[0:4]
            chunk_size = struct.unpack("<I", chunk_header[4:8])[0]
            if chunk_id == b"fmt ":
                fmt = handle.read(chunk_size)
                channels = struct.unpack("<H", fmt[2:4])[0]
                sample_rate = struct.unpack("<I", fmt[4:8])[0]
                bits_per_sample = struct.unpack("<H", fmt[14:16])[0]
            elif chunk_id == b"data":
                data_bytes = chunk_size
                break
            else:
                # Skip this chunk (chunks are word-aligned, so pad odd sizes).
                handle.seek(chunk_size + (chunk_size & 1), 1)
        if None in (channels, sample_rate, bits_per_sample, data_bytes):
            raise ValueError("missing fmt/data chunk fields")
        frames = data_bytes // (channels * (bits_per_sample // 8))
        return frames / float(sample_rate)


def resolve_audio_path(stored_path: str, data_root: Path) -> Path:
    """Remap a stored absolute audio path onto the local NISQA corpus root.

    The SFT records store cluster paths under ``.../NISQA_Corpus/...``; rebase
    everything after ``NISQA_Corpus/`` onto the local ``data_root``.

    Args:
        stored_path: Audio path as stored in the SFT record.
        data_root: Local ``data/raw/NISQA_Corpus`` directory.

    Returns:
        Local filesystem path to the clip.
    """
    marker = "NISQA_Corpus/"
    if marker in stored_path:
        return data_root / stored_path.split(marker, 1)[1]
    return Path(stored_path)


def measure_durations(
    data_frame: pd.DataFrame, data_root: Path, sample_limit: int | None
) -> np.ndarray:
    """Read clip durations from the WAV RIFF headers only (no sample decode).

    Args:
        data_frame: Joined dataframe with a ``stored_audio`` column.
        data_root: Local ``data/raw/NISQA_Corpus`` directory.
        sample_limit: Optional cap on the number of files probed.

    Returns:
        Array of clip durations in seconds.
    """
    stored = data_frame["stored_audio"].tolist()
    if sample_limit is not None and len(stored) > sample_limit:
        step = len(stored) / sample_limit
        stored = [stored[int(i * step)] for i in range(sample_limit)]
        print(f"Duration: sampling {len(stored)} of {len(data_frame)} clip headers.")
    else:
        print(f"Duration: reading headers for all {len(stored)} clips.")

    durations: list[float] = []
    missing = 0
    for stored_path in stored:
        wav_path = resolve_audio_path(stored_path, data_root)
        try:
            durations.append(wav_duration_seconds(wav_path))
        except Exception:
            missing += 1
    if missing:
        print(f"Duration: {missing} files could not be read.")
    return np.asarray(durations, dtype=float)


def annotate_bars(axis: plt.Axes, bars, values, fmt: str = "{:.0f}") -> None:
    """Write a count label above each bar."""
    for rect, value in zip(bars, values):
        axis.annotate(
            fmt.format(value),
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#333333",
        )


def save(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    """Save a figure as both PDF and PNG."""
    for ext in ("pdf", "png"):
        out = figures_dir / f"{stem}.{ext}"
        fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def fig_mos_distribution(data_frame: pd.DataFrame, figures_dir: Path) -> dict:
    """Figure 1: MOS distribution over the full training set."""
    mos = data_frame["mos"].to_numpy()
    fig, axis = plt.subplots(figsize=(6.0, 3.4))
    bins = np.arange(1.0, 5.0 + 0.2, 0.2)
    axis.hist(mos, bins=bins, color=ACCENT, edgecolor="white", linewidth=0.5)

    mean_mos = float(np.mean(mos))
    median_mos = float(np.median(mos))
    axis.axvline(mean_mos, color="#c0392b", linewidth=1.3, linestyle="--",
                 label=f"mean = {mean_mos:.2f}")
    axis.axvline(median_mos, color="#222222", linewidth=1.1, linestyle=":",
                 label=f"median = {median_mos:.2f}")

    axis.set_xlabel("Mean Opinion Score (MOS)")
    axis.set_ylabel("number of clips")
    axis.set_title("MOS distribution of the SFT training set (n = "
                   f"{len(mos):,})")
    axis.set_xlim(1.0, 5.0)
    axis.set_xticks(np.arange(1.0, 5.5, 0.5))
    axis.legend(loc="upper right")
    axis.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, figures_dir, "eda_mos_distribution")

    return {
        "n": int(len(mos)),
        "min": float(np.min(mos)),
        "max": float(np.max(mos)),
        "mean": mean_mos,
        "median": median_mos,
        "std": float(np.std(mos)),
        "frac_le_3": float(np.mean(mos <= 3.0)),
        "frac_gt_3": float(np.mean(mos > 3.0)),
    }


def fig_degradation_taxonomy(data_frame: pd.DataFrame, figures_dir: Path) -> dict:
    """Figure 2: 13 raw tags vs the 6-category collapse, side by side."""
    raw_counts = collections.Counter()
    for tags in data_frame["active_tags"]:
        for tag in tags:
            raw_counts[tag] += 1
    cat_counts = collections.Counter()
    for tag, count in raw_counts.items():
        cat_counts[CATEGORY_MAP[tag]] += count

    fig, (ax_raw, ax_cat) = plt.subplots(
        1, 2, figsize=(9.2, 4.0), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )

    # Left: 13 raw tags, coloured by the family they collapse into.
    raw_order = DEGRADATION_COLUMNS
    raw_vals = [raw_counts.get(tag, 0) for tag in raw_order]
    raw_colors = [CATEGORY_COLORS[CATEGORY_MAP[tag]] for tag in raw_order]
    bars_raw = ax_raw.bar(range(len(raw_order)), raw_vals, color=raw_colors,
                          edgecolor="white", linewidth=0.5)
    ax_raw.set_xticks(range(len(raw_order)))
    ax_raw.set_xticklabels(raw_order, rotation=45, ha="right", fontsize=8)
    ax_raw.set_ylabel("clips containing tag")
    ax_raw.set_title("(a) 13 raw NISQA degradation tags")
    ax_raw.grid(axis="x", visible=False)
    annotate_bars(ax_raw, bars_raw, raw_vals)

    # Right: 6 collapsed categories.
    cat_order = [c for c in CATEGORY_ORDER if c in cat_counts]
    cat_vals = [cat_counts[c] for c in cat_order]
    cat_colors = [CATEGORY_COLORS[c] for c in cat_order]
    bars_cat = ax_cat.bar(range(len(cat_order)), cat_vals, color=cat_colors,
                          edgecolor="white", linewidth=0.5)
    ax_cat.set_xticks(range(len(cat_order)))
    ax_cat.set_xticklabels(
        [c.replace(" ", "\n") for c in cat_order], fontsize=8
    )
    ax_cat.set_ylabel("clips containing category")
    ax_cat.set_title("(b) 6 collapsed categories")
    ax_cat.grid(axis="x", visible=False)
    annotate_bars(ax_cat, bars_cat, cat_vals)

    fig.suptitle(
        "Degradation taxonomy: the 13 NISQA tags collapse to 6 legible categories",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    save(fig, figures_dir, "eda_degradation_taxonomy")

    return {
        "raw_tag_counts": {tag: int(raw_counts.get(tag, 0)) for tag in raw_order},
        "category_counts": {c: int(cat_counts[c]) for c in cat_order},
        "total_tag_instances": int(sum(raw_vals)),
        "smallest_raw_tag": min(raw_order, key=lambda t: raw_counts.get(t, 0)),
        "smallest_raw_tag_count": int(min(raw_vals)),
        "largest_category": cat_order[0],
        "largest_category_count": int(cat_vals[0]),
    }


def fig_degradations_per_clip(data_frame: pd.DataFrame, figures_dir: Path) -> dict:
    """Figure 3: how many simultaneous degradations each clip carries."""
    num_tags = data_frame["num_tags"].to_numpy()
    counts = collections.Counter(num_tags.tolist())
    max_tags = int(num_tags.max())
    # Start at 0 so the clean (zero-tag) clips are shown, not silently dropped.
    xs = list(range(0, max_tags + 1))
    ys = [counts.get(x, 0) for x in xs]

    fig, axis = plt.subplots(figsize=(6.0, 3.4))
    colors = ["#9aa7b1"] + [ACCENT] * (len(xs) - 1)
    bars = axis.bar(xs, ys, color=colors, edgecolor="white", linewidth=0.5)
    annotate_bars(axis, bars, ys)
    mean_tags = float(np.mean(num_tags))
    axis.axvline(mean_tags, color="#c0392b", linewidth=1.2, linestyle="--",
                 label=f"mean = {mean_tags:.2f} tags / clip")
    axis.set_xlabel("number of simultaneous degradation tags on a clip")
    axis.set_ylabel("number of clips")
    axis.set_title("Most clips carry several degradations at once")
    axis.set_xticks(xs)
    axis.legend(loc="upper right")
    axis.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, figures_dir, "eda_degradations_per_clip")

    multi = float(np.mean(num_tags >= 2))
    return {
        "counts": {int(k): int(v) for k, v in sorted(counts.items())},
        "mean_tags_per_clip": mean_tags,
        "frac_multi_tag": multi,
        "frac_single_tag": float(np.mean(num_tags == 1)),
        "frac_zero_tag": float(np.mean(num_tags == 0)),
        "max_tags": max_tags,
    }


def fig_clip_duration(durations: np.ndarray, figures_dir: Path) -> dict:
    """Figure 4: clip duration distribution in seconds."""
    fig, axis = plt.subplots(figsize=(6.0, 3.4))
    if durations.size:
        upper = float(np.percentile(durations, 99.5))
        bins = np.linspace(0, max(upper, 1.0), 40)
        axis.hist(durations, bins=bins, color=ACCENT, edgecolor="white",
                  linewidth=0.5)
        mean_d = float(np.mean(durations))
        median_d = float(np.median(durations))
        axis.axvline(mean_d, color="#c0392b", linewidth=1.3, linestyle="--",
                     label=f"mean = {mean_d:.2f} s")
        axis.axvline(median_d, color="#222222", linewidth=1.1, linestyle=":",
                     label=f"median = {median_d:.2f} s")
        axis.legend(loc="upper right")
    else:
        mean_d = median_d = 0.0
    axis.set_xlabel("clip duration (seconds)")
    axis.set_ylabel("number of clips")
    axis.set_title(f"Clip duration distribution (n = {durations.size:,})")
    axis.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, figures_dir, "eda_clip_duration")

    if not durations.size:
        return {"n": 0}
    return {
        "n": int(durations.size),
        "min": float(np.min(durations)),
        "max": float(np.max(durations)),
        "mean": mean_d,
        "median": median_d,
        "std": float(np.std(durations)),
        "p05": float(np.percentile(durations, 5)),
        "p95": float(np.percentile(durations, 95)),
    }


def fig_source_composition(data_frame: pd.DataFrame, figures_dir: Path) -> dict:
    """Figure 5: source-corpus composition and MOS spread per source."""
    source_counts = data_frame["source"].value_counts()
    order = source_counts.index.tolist()

    fig, (ax_count, ax_mos) = plt.subplots(
        1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.0, 1.1]}
    )

    # Left: clip count per source corpus.
    bars = ax_count.bar(range(len(order)), source_counts.values, color=ACCENT,
                        edgecolor="white", linewidth=0.5)
    annotate_bars(ax_count, bars, source_counts.values)
    ax_count.set_xticks(range(len(order)))
    ax_count.set_xticklabels(order, fontsize=8)
    ax_count.set_ylabel("number of clips")
    ax_count.set_title("(a) clips per source corpus")
    ax_count.grid(axis="x", visible=False)

    # Right: MOS distribution per source corpus (boxplot).
    mos_by_source = [data_frame.loc[data_frame["source"] == s, "mos"].to_numpy()
                     for s in order]
    box = ax_mos.boxplot(
        mos_by_source, vert=True, patch_artist=True, widths=0.6,
        medianprops={"color": "#222222", "linewidth": 1.2},
        flierprops={"marker": ".", "markersize": 3,
                    "markerfacecolor": "#999999", "markeredgecolor": "none"},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(ACCENT)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#444444")
    ax_mos.set_xticks(range(1, len(order) + 1))
    ax_mos.set_xticklabels(order, fontsize=8)
    ax_mos.set_ylabel("MOS")
    ax_mos.set_ylim(1.0, 5.0)
    ax_mos.set_title("(b) MOS spread per source corpus")
    ax_mos.grid(axis="x", visible=False)

    fig.suptitle("Source-corpus composition of the SFT training set",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    save(fig, figures_dir, "eda_source_composition")

    return {
        "counts": {s: int(source_counts[s]) for s in order},
        "mos_mean_by_source": {
            s: float(data_frame.loc[data_frame["source"] == s, "mos"].mean())
            for s in order
        },
    }


def main() -> None:
    """Run the full descriptive analysis and write figures plus a stats JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sft-jsonl",
        type=Path,
        default=Path("data/processed/sft/train_nisqa_llama_10k.json"),
        help="Global MOS SFT training set (JSONL).",
    )
    parser.add_argument(
        "--nisqa-csv",
        type=Path,
        default=Path(
            "data/raw/NISQA_Corpus/NISQA_TRAIN_SIM/NISQA_TRAIN_SIM_file.csv"
        ),
        help="NISQA per-file metadata CSV used to recover tags and source.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/NISQA_Corpus"),
        help="Root for resolving NISQA filepath_deg entries when reading durations.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("../_papers/asa-thesis/figures"),
        help="Where to write the figures (defaults to the thesis figures dir).",
    )
    parser.add_argument(
        "--duration-sample",
        type=int,
        default=None,
        help="Optional cap on clips probed for duration (header reads only).",
    )
    args = parser.parse_args()

    figures_dir = args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    print(f"Loading SFT records from {args.sft_jsonl} ...")
    records = load_records(args.sft_jsonl)
    print(f"  {len(records):,} records")

    print(f"Joining to {args.nisqa_csv} ...")
    data_frame = build_dataframe(records, args.nisqa_csv)
    print(f"  joined {len(data_frame):,} records")

    print("Reading clip durations ...")
    durations = measure_durations(data_frame, args.data_root, args.duration_sample)

    print(f"Writing figures to {figures_dir} ...")
    stats = {
        "dataset": str(args.sft_jsonl),
        "n_records": int(len(data_frame)),
        "mos_distribution": fig_mos_distribution(data_frame, figures_dir),
        "degradation_taxonomy": fig_degradation_taxonomy(data_frame, figures_dir),
        "degradations_per_clip": fig_degradations_per_clip(data_frame, figures_dir),
        "clip_duration": fig_clip_duration(durations, figures_dir),
        "source_composition": fig_source_composition(data_frame, figures_dir),
    }

    stats_path = figures_dir / "eda_nisqa_mos_10k_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote stats to {stats_path}")
    print("\nKey numbers:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
