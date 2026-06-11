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
import seaborn as sns


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

# --- Shared visual identity -------------------------------------------------
# A single professional palette drives every figure so the chapter reads as one
# coherent set. PRIMARY is a deep teal used for the main distributions; HIGHLIGHT
# is a warm amber reserved for mean/median reference lines; MUTED greys the
# "no degradation" baseline. The six-colour qualitative palette is a curated,
# harmonious set (deep teal, amber, sage, plum, terracotta, slate) reused for the
# degradation categories so a colour always means the same family.

PRIMARY = "#2a6f7f"        # deep teal: main bars / histograms
PRIMARY_DARK = "#1f5360"   # darker teal: emphasis / edges
HIGHLIGHT = "#d98b34"      # warm amber: mean reference line
HIGHLIGHT_2 = "#3d3d3d"    # near-black: secondary (median) reference line
MUTED = "#b7c4c7"          # soft grey-teal: baseline / zero category

# Six-colour qualitative palette for the degradation categories (harmonious,
# colour-blind-friendlier than the old primary/secondary clash).
CATEGORY_PALETTE: list[str] = [
    "#2a6f7f",  # deep teal
    "#d98b34",  # amber
    "#6a994e",  # sage green
    "#8e6c9b",  # plum
    "#c1574b",  # terracotta
    "#5c6b73",  # slate
]

# One colour per collapsed category; raw tags inherit their family colour so the
# two panels of the taxonomy figure read as the same grouping.
CATEGORY_COLORS: dict[str, str] = {
    category: CATEGORY_PALETTE[index]
    for index, category in enumerate(CATEGORY_ORDER)
}

# Back-compat aliases used throughout the figure functions.
ACCENT = PRIMARY
GRID_GREY = "#d9d9d9"


def set_paper_style() -> None:
    """Apply one cohesive seaborn-based, publication-oriented theme.

    Every figure in the chapter shares this theme so they read as a single set:
    seaborn's ``whitegrid`` base for soft horizontal guides, a serif font to match
    the thesis body text, despined axes, and the shared qualitative palette as the
    default colour cycle.
    """
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette=CATEGORY_PALETTE,
        font="serif",
    )
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.labelcolor": "#222222",
            "axes.edgecolor": "#5c6b73",
            "axes.linewidth": 0.9,
            "text.color": "#222222",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "legend.fontsize": 9,
            "legend.frameon": False,
            "grid.color": GRID_GREY,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "axes.axisbelow": True,
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
                "response": str(record.get("response", "")),
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
    fig, axis = plt.subplots(figsize=(6.0, 3.0))
    sns.histplot(
        x=mos,
        bins=np.arange(0.9, 5.11, 0.2),
        stat="percent",
        color="#999999",
        edgecolor="white",
        linewidth=0.6,
        alpha=1.0,
        ax=axis,
    )
    quality_scale = plt.get_cmap("viridis")
    for bar in axis.patches:
        center = bar.get_x() + bar.get_width() / 2
        bar.set_facecolor(quality_scale((center - 1.0) / 4.0))

    mean_mos = float(np.mean(mos))
    median_mos = float(np.median(mos))
    axis.axvline(
        mean_mos,
        color="#252525",
        linewidth=1.5,
        linestyle="--",
        zorder=3,
    )
    fig.text(
        0.98,
        0.965,
        f"Mean {mean_mos:.2f}   |   Median {median_mos:.2f}",
        ha="right",
        va="top",
        fontsize=9,
        color="#333333",
    )

    axis.set_xlabel("Mean Opinion Score (MOS)")
    axis.set_ylabel("Share of Clips (%)")
    axis.set_xlim(0.9, 5.1)
    axis.set_xticks(np.arange(1.0, 5.5, 0.5))
    axis.grid(axis="x", visible=False)
    sns.despine(ax=axis)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
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
    colors = [MUTED] + [ACCENT] * (len(xs) - 1)
    bars = axis.bar(xs, ys, color=colors, edgecolor="white", linewidth=0.6)
    annotate_bars(axis, bars, ys)
    mean_tags = float(np.mean(num_tags))
    axis.axvline(mean_tags, color=HIGHLIGHT, linewidth=1.6, linestyle="--",
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
        axis.axvline(mean_d, color=HIGHLIGHT, linewidth=1.6, linestyle="--",
                     label=f"mean = {mean_d:.2f} s")
        axis.axvline(median_d, color=HIGHLIGHT_2, linewidth=1.2, linestyle=":",
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

    # Left: clip count per source corpus, one palette colour per corpus so the
    # two panels share a per-source colour identity.
    count_colors = [CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)]
                    for i in range(len(order))]
    bars = ax_count.bar(range(len(order)), source_counts.values, color=count_colors,
                        edgecolor="white", linewidth=0.6)
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
        mos_by_source, vert=True, patch_artist=True, widths=0.62,
        medianprops={"color": "white", "linewidth": 1.5},
        whiskerprops={"color": "#5c6b73", "linewidth": 1.1},
        capprops={"color": "#5c6b73", "linewidth": 1.1},
        flierprops={"marker": ".", "markersize": 3,
                    "markerfacecolor": "#999999", "markeredgecolor": "none"},
    )
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(CATEGORY_PALETTE[index % len(CATEGORY_PALETTE)])
        patch.set_alpha(0.9)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.8)
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


def caption_words(text: str) -> list[str]:
    """Tokenize a caption into lowercase word tokens."""
    import re

    return re.findall(r"[A-Za-z']+", text)


def fig_caption_length(data_frame: pd.DataFrame, figures_dir: Path) -> dict:
    """Caption length distribution and the length-vs-MOS relationship.

    The caption is the literal supervision target, and its length couples with
    quality, which is worth flagging next to a length-sensitive metric like BLEU.
    """
    responses = data_frame["response"].tolist()
    mos = data_frame["mos"].to_numpy()
    word_counts = np.array([len(caption_words(text)) for text in responses])

    pearson = float(np.corrcoef(word_counts, mos)[0, 1])

    fig, (ax_hist, ax_scatter) = plt.subplots(
        1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.0, 1.1]}
    )

    # Left: caption-length histogram.
    bins = np.arange(word_counts.min(), word_counts.max() + 2)
    ax_hist.hist(word_counts, bins=bins, color=ACCENT, edgecolor="white",
                 linewidth=0.4)
    mean_w = float(np.mean(word_counts))
    median_w = float(np.median(word_counts))
    ax_hist.axvline(mean_w, color=HIGHLIGHT, linewidth=1.6, linestyle="--",
                    label=f"mean = {mean_w:.1f}")
    ax_hist.axvline(median_w, color=HIGHLIGHT_2, linewidth=1.2, linestyle=":",
                    label=f"median = {median_w:.0f}")
    ax_hist.set_xlabel("caption length (words)")
    ax_hist.set_ylabel("number of captions")
    ax_hist.set_title("(a) caption-length distribution")
    ax_hist.legend(loc="upper right")
    ax_hist.grid(axis="x", visible=False)

    # Right: mean caption length per MOS band, with the Pearson r annotated.
    band_edges = [1.0, 2.0, 3.0, 4.0, 5.01]
    band_labels = ["1-2", "2-3", "3-4", "4-5"]
    band_means: list[float] = []
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        mask = (mos >= low) & (mos < high)
        band_means.append(float(np.mean(word_counts[mask])) if mask.any() else 0.0)
    bars = ax_scatter.bar(range(len(band_labels)), band_means, color=ACCENT,
                          edgecolor="white", linewidth=0.5)
    annotate_bars(ax_scatter, bars, band_means, fmt="{:.1f}")
    ax_scatter.set_xticks(range(len(band_labels)))
    ax_scatter.set_xticklabels(band_labels)
    ax_scatter.set_xlabel("MOS band")
    ax_scatter.set_ylabel("mean caption length (words)")
    ax_scatter.set_title(f"(b) length rises with MOS (Pearson r = {pearson:.2f})")
    ax_scatter.grid(axis="x", visible=False)

    fig.suptitle("Caption length and its coupling with MOS", fontsize=11, y=1.02)
    fig.tight_layout()
    save(fig, figures_dir, "eda_caption_length")

    return {
        "words_min": int(word_counts.min()),
        "words_max": int(word_counts.max()),
        "words_mean": mean_w,
        "words_median": median_w,
        "words_std": float(np.std(word_counts)),
        "pearson_r_words_mos": pearson,
        "mean_words_by_mos_band": dict(zip(band_labels, [round(v, 1) for v in band_means])),
    }


def fig_caption_templating(data_frame: pd.DataFrame, figures_dir: Path) -> dict:
    """Caption templating: vocabulary size, distinct-n, and dominant openings.

    99% of captions are technically unique, but they are slot-filled from a tiny
    vocabulary on a fixed scaffold. This is the output-side honesty check and the
    main caveat on BLEU.
    """
    responses = data_frame["response"].tolist()
    n = len(responses)
    unique = len({r for r in responses})

    all_tokens = [w.lower() for text in responses for w in caption_words(text)]
    vocab = len(set(all_tokens))

    def distinct_n(tokens: list[str], order: int) -> float:
        grams = [tuple(tokens[i:i + order]) for i in range(len(tokens) - order + 1)]
        return len(set(grams)) / len(grams) if grams else 0.0

    distinct = {k: distinct_n(all_tokens, k) for k in (1, 2, 3)}

    openings = collections.Counter(
        " ".join(caption_words(text)[:5]).lower() for text in responses
    )
    top_openings = openings.most_common(6)

    fig, axis = plt.subplots(figsize=(7.4, 3.8))
    labels = [f'"{phrase}…"' for phrase, _ in top_openings]
    values = [count for _, count in top_openings]
    y_pos = range(len(labels))
    bars = axis.barh(list(y_pos), values, color=ACCENT, edgecolor="white",
                     linewidth=0.5)
    for rect, value in zip(bars, values):
        axis.annotate(f"{100 * value / n:.1f}%",
                      xy=(rect.get_width(), rect.get_y() + rect.get_height() / 2),
                      xytext=(3, 0), textcoords="offset points",
                      va="center", fontsize=8, color="#333333")
    axis.set_yticks(list(y_pos))
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("number of captions")
    axis.set_title(
        f"Captions are slot-filled from a {vocab}-word vocabulary "
        f"(distinct-1 = {distinct[1]:.4f})"
    )
    axis.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, figures_dir, "eda_caption_templating")

    return {
        "n": n,
        "unique_captions": unique,
        "unique_fraction": unique / n,
        "exact_duplicates": n - unique,
        "vocabulary_size": vocab,
        "total_tokens": len(all_tokens),
        "distinct_1": distinct[1],
        "distinct_2": distinct[2],
        "distinct_3": distinct[3],
        "top_openings": [
            {"opening": phrase, "count": count, "fraction": count / n}
            for phrase, count in top_openings
        ],
    }


def analyse_quality_vocabulary(data_frame: pd.DataFrame) -> dict:
    """Frequency of descriptive quality terms across captions (table source)."""
    responses = [text.lower() for text in data_frame["response"].tolist()]
    n = len(responses)
    terms = [
        "distortion", "discontinu", "noise", "noisy", "loud", "volume",
        "clear", "clean", "natural", "unnatural", "soft", "quiet",
        "background", "distorted",
    ]
    counts = {term: sum(1 for text in responses if term in text) for term in terms}
    ordered = sorted(counts.items(), key=lambda item: -item[1])
    return {
        "n": n,
        "term_counts": {term: count for term, count in ordered if count > 0},
    }


def analyse_mos_consistency(data_frame: pd.DataFrame) -> dict:
    """Check that the MOS stated inside the caption matches the mos label."""
    import re

    responses = data_frame["response"].tolist()
    mos = data_frame["mos"].tolist()
    n = len(responses)
    pattern = re.compile(
        r"MOS\s*(?:score)?\s*(?:is|of|:)?\s*(?:only\s*)?(\d(?:\.\d+)?)", re.I
    )
    mentions = parsed = agree = disagree = 0
    for text, label in zip(responses, mos):
        if re.search(r"\bMOS\b", text, re.I):
            mentions += 1
        match = pattern.search(text)
        if match:
            parsed += 1
            try:
                if abs(float(match.group(1)) - float(label)) <= 0.05:
                    agree += 1
                else:
                    disagree += 1
            except ValueError:
                pass
    return {
        "n": n,
        "mention_mos": mentions,
        "parsed_numeric": parsed,
        "agree": agree,
        "disagree": disagree,
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
        "caption_length": fig_caption_length(data_frame, figures_dir),
        "caption_templating": fig_caption_templating(data_frame, figures_dir),
        "quality_vocabulary": analyse_quality_vocabulary(data_frame),
        "mos_consistency": analyse_mos_consistency(data_frame),
    }

    stats_path = figures_dir / "eda_nisqa_mos_10k_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote stats to {stats_path}")
    print("\nKey numbers:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
