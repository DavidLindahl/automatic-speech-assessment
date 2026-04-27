"""Generate NISQA-SIM low-MOS temporal mixes with one active-region degradation segment.

This script converts NISQA REF/DEG pairs into mixed files where each output clip
contains exactly one inserted degradation region.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import typer
from tqdm import tqdm


@dataclass(frozen=True)
class Segment:
    """Represents one degradation interval in seconds.

    Attributes:
        start: Segment start time in seconds.
        end: Segment end time in seconds.
    """

    start: float
    end: float


@dataclass(frozen=True)
class PlacementConfig:
    """Stores active-region placement settings.

    Attributes:
        segment_min_seconds: Minimum segment duration.
        segment_max_ratio: Maximum segment duration as ratio of clip duration.
        min_active_fraction: Minimum active fraction required inside chosen segment.
        activity_std_multiplier: Multiplier on waveform standard deviation for activity threshold.
        activity_abs_floor: Absolute floor for the activity threshold.
        activity_smooth_ms: Smoothing window size for activity envelope.
        segment_attempts: Number of stochastic segment sampling attempts per row.
        energy_quantile: Quantile cutoff for candidate window energy.
        energy_top_k_ratio: Ratio of top-energy candidates used as sampling pool.
        output_active_fraction_min: Minimum active fraction accepted for final output.
        max_row_attempts: Number of segment placement retries for the same source pair.
    """

    segment_min_seconds: float
    segment_max_ratio: float
    min_active_fraction: float
    activity_std_multiplier: float
    activity_abs_floor: float
    activity_smooth_ms: int
    segment_attempts: int
    energy_quantile: float
    energy_top_k_ratio: float
    output_active_fraction_min: float
    max_row_attempts: int


DEGRADATION_COLUMNS = [
    "filter",
    "timeclipping",
    "wbgn",
    "p50mnru",
    "bgn",
    "clipping",
    "arb_filter",
    "codec1",
    "codec2",
    "codec3",
    "plcMode1",
    "plcMode2",
    "plcMode3",
]


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load audio and return mono float32 waveform with sample rate.

    Args:
        path: Path to the WAV file.

    Returns:
        Tuple of mono waveform and input sample rate.
    """

    audio, sample_rate = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), int(sample_rate)


def resample_if_needed(
    audio: np.ndarray, sample_rate_in: int, sample_rate_out: int
) -> np.ndarray:
    """Resample waveform when source and target sample rates differ.

    Args:
        audio: Input mono waveform.
        sample_rate_in: Input sample rate.
        sample_rate_out: Target sample rate.

    Returns:
        Resampled waveform.
    """

    if sample_rate_in == sample_rate_out:
        return audio
    gcd = math.gcd(sample_rate_in, sample_rate_out)
    up = sample_rate_out // gcd
    down = sample_rate_in // gcd
    return resample_poly(audio, up=up, down=down).astype(np.float32)


def align_pair(
    ref_audio: np.ndarray,
    deg_audio: np.ndarray,
    target_sample_rate: int,
    max_duration_seconds: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Align REF/DEG waveforms to a shared length without padding.

    Args:
        ref_audio: Reference waveform.
        deg_audio: Degraded waveform.
        target_sample_rate: Target sample rate used for max-duration truncation.
        max_duration_seconds: Optional maximum clip duration.

    Returns:
        Length-aligned REF and DEG waveforms.
    """

    length = min(len(ref_audio), len(deg_audio))
    if max_duration_seconds is not None:
        max_len = int(round(max_duration_seconds * target_sample_rate))
        length = min(length, max_len)
    return ref_audio[:length], deg_audio[:length]


def is_active_tag(value: object) -> bool:
    """Return whether a metadata cell encodes an active degradation tag.

    Args:
        value: Cell value from a degradation metadata column.

    Returns:
        True when the value indicates an active degradation, else False.
    """

    if pd.isna(value):
        return False
    token = str(value).strip()
    return token not in {"", "-", "nan", "None"}


def extract_active_degradations(row: pd.Series) -> list[str]:
    """Extract active degradation labels from one NISQA metadata row.

    Args:
        row: One row from the NISQA per-file CSV.

    Returns:
        List of active degradation column names.
    """

    return [
        column
        for column in DEGRADATION_COLUMNS
        if is_active_tag(row.get(column, np.nan))
    ]


def smooth_envelope(audio: np.ndarray, sample_rate: int, smooth_ms: int) -> np.ndarray:
    """Compute a smoothed absolute-amplitude envelope.

    Args:
        audio: Mono waveform.
        sample_rate: Sample rate.
        smooth_ms: Smoothing window size in milliseconds.

    Returns:
        Smoothed envelope.
    """

    window = max(1, int(round(sample_rate * (smooth_ms / 1000.0))))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(np.abs(audio), kernel, mode="same").astype(np.float32)


def window_energy_sums(signal: np.ndarray, window: int) -> np.ndarray:
    """Compute moving window sums.

    Args:
        signal: Input signal.
        window: Window size in samples.

    Returns:
        Array of moving sums with length ``len(signal) - window + 1``.
    """

    if window <= 1:
        return signal.copy()
    if len(signal) < window:
        return np.array([], dtype=np.float32)
    cumsum = np.concatenate(([0.0], np.cumsum(signal, dtype=np.float64)))
    sums = cumsum[window:] - cumsum[:-window]
    return sums.astype(np.float32)


def choose_active_segment(
    ref_audio: np.ndarray,
    sample_rate: int,
    config: PlacementConfig,
    rng: random.Random,
) -> tuple[Segment, float, float]:
    """Choose one active-region segment using high-energy window ranking.

    Args:
        ref_audio: Reference waveform.
        sample_rate: Waveform sample rate.
        config: Segment placement configuration.
        rng: Random generator instance.

    Returns:
        Segment, activity threshold, and active fraction in that segment.
    """

    n_samples = len(ref_audio)
    total_seconds = n_samples / float(sample_rate)
    if n_samples < 2:
        return Segment(0.0, 0.0), config.activity_abs_floor, 0.0

    envelope = smooth_envelope(ref_audio, sample_rate, config.activity_smooth_ms)
    threshold = max(
        config.activity_abs_floor,
        config.activity_std_multiplier * float(np.std(ref_audio)),
    )
    max_seg_seconds = max(
        config.segment_min_seconds, total_seconds * config.segment_max_ratio
    )

    def sample_segment_samples() -> int:
        segment_seconds = rng.uniform(config.segment_min_seconds, max_seg_seconds)
        segment_samples = max(1, int(round(segment_seconds * sample_rate)))
        return min(segment_samples, max(1, n_samples - 1))

    for _ in range(config.segment_attempts):
        seg_samples = sample_segment_samples()
        energy = window_energy_sums(envelope, seg_samples)
        if len(energy) == 0:
            continue

        valid_length = len(energy)
        energy_cutoff = float(np.quantile(energy, config.energy_quantile))
        starts = np.arange(valid_length)
        candidates = starts[
            (energy >= energy_cutoff) & (envelope[:valid_length] >= threshold)
        ]
        if len(candidates) == 0:
            continue

        candidate_energy = energy[candidates]
        top_k = max(1, int(math.ceil(len(candidates) * config.energy_top_k_ratio)))
        top_indices = np.argsort(candidate_energy)[-top_k:]
        pool = candidates[top_indices]

        start_idx = int(pool[rng.randrange(len(pool))])
        end_idx = min(n_samples, start_idx + seg_samples)
        if end_idx <= start_idx:
            continue

        active_fraction = float(np.mean(envelope[start_idx:end_idx] >= threshold))
        if active_fraction >= config.min_active_fraction:
            return (
                Segment(start_idx / sample_rate, end_idx / sample_rate),
                threshold,
                active_fraction,
            )

    fallback_seg = max(
        1,
        int(
            round(
                max(
                    config.segment_min_seconds,
                    min(max_seg_seconds, total_seconds * 0.33),
                )
                * sample_rate
            )
        ),
    )
    fallback_seg = min(fallback_seg, max(1, n_samples - 1))

    energy = window_energy_sums(envelope, fallback_seg)
    if len(energy) == 0:
        return (
            Segment(0.0, min(total_seconds, config.segment_min_seconds)),
            threshold,
            0.0,
        )

    valid_length = len(energy)
    starts = np.arange(valid_length)
    start_mask = envelope[:valid_length] >= threshold
    if np.any(start_mask):
        weighted = np.where(start_mask, energy, -np.inf)
        best_start = int(np.argmax(weighted))
    else:
        best_start = int(np.argmax(energy))

    start_idx = int(starts[best_start])
    end_idx = min(n_samples, start_idx + fallback_seg)
    if end_idx <= start_idx:
        return (
            Segment(0.0, min(total_seconds, config.segment_min_seconds)),
            threshold,
            0.0,
        )
    active_fraction = float(np.mean(envelope[start_idx:end_idx] >= threshold))
    return (
        Segment(start_idx / sample_rate, end_idx / sample_rate),
        threshold,
        active_fraction,
    )


def build_mix_one_segment(
    ref_audio: np.ndarray,
    deg_audio: np.ndarray,
    sample_rate: int,
    config: PlacementConfig,
    rng: random.Random,
) -> tuple[np.ndarray, Segment, float, float]:
    """Create one mix with exactly one inserted degradation segment.

    Args:
        ref_audio: Reference waveform.
        deg_audio: Degraded waveform.
        sample_rate: Waveform sample rate.
        config: Segment placement configuration.
        rng: Random generator instance.

    Returns:
        Mixed waveform, chosen segment, threshold, and active fraction.
    """

    segment, threshold, active_fraction = choose_active_segment(
        ref_audio=ref_audio,
        sample_rate=sample_rate,
        config=config,
        rng=rng,
    )
    mixed = ref_audio.copy()
    i0 = max(0, min(int(round(segment.start * sample_rate)), len(mixed)))
    i1 = max(i0, min(int(round(segment.end * sample_rate)), len(mixed)))
    mixed[i0:i1] = deg_audio[i0:i1]
    return mixed, segment, threshold, active_fraction


def serialize_segments(segments: list[Segment]) -> str:
    """Serialize segment list as JSON.

    Args:
        segments: List of segments.

    Returns:
        JSON-serialized segment list.
    """

    payload = [
        {"start": round(seg.start, 3), "end": round(seg.end, 3)} for seg in segments
    ]
    return json.dumps(payload)


def timeline_from_segment(
    duration_seconds: float, segment: Segment
) -> list[dict[str, float | str]]:
    """Build REF/DEG timeline from one degradation segment.

    Args:
        duration_seconds: Full clip duration.
        segment: Degradation segment.

    Returns:
        Timeline entries with source labels.
    """

    timeline: list[dict[str, float | str]] = []
    if segment.start > 0.0:
        timeline.append({"start": 0.0, "end": round(segment.start, 3), "source": "REF"})
    timeline.append(
        {
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "source": "DEG",
        }
    )
    if segment.end < duration_seconds:
        timeline.append(
            {
                "start": round(segment.end, 3),
                "end": round(duration_seconds, 3),
                "source": "REF",
            }
        )
    return timeline


def switch_points_from_timeline(timeline: list[dict[str, float | str]]) -> list[float]:
    """Extract unique boundary points from timeline entries.

    Args:
        timeline: Timeline entries.

    Returns:
        Sorted unique switch points.
    """

    points: list[float] = []
    for item in timeline:
        points.extend([float(item["start"]), float(item["end"])])
    return sorted(set(round(point, 3) for point in points))


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Create or validate output directory state.

    Args:
        output_dir: Output directory.
        overwrite: Whether existing files should be removed.

    Raises:
        ValueError: If output exists and overwrite is False.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_mix_files = list(output_dir.glob("*_mix_*.wav"))
    manifest_path = output_dir / "manifest.csv"
    has_existing = bool(existing_mix_files) or manifest_path.exists()

    if has_existing and not overwrite:
        raise ValueError(
            f"Output directory {output_dir} already contains generated files. "
            "Use --overwrite to replace them."
        )

    if overwrite:
        for file_path in existing_mix_files:
            file_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()


def load_eligible_rows(
    data_root: Path,
    sim_split: str,
    mos_max_threshold: float,
    require_active_degradation_types: bool,
) -> pd.DataFrame:
    """Load and filter NISQA rows eligible for mix generation.

    Args:
        data_root: Root path to NISQA corpus.
        sim_split: Split name, for example ``NISQA_TRAIN_SIM``.
        mos_max_threshold: Maximum MOS allowed in source rows.
        require_active_degradation_types: Whether rows must have active degradation tags.

    Returns:
        Filtered dataframe with extra helper columns.
    """

    csv_path = data_root / sim_split / f"{sim_split}_file.csv"
    data_frame = pd.read_csv(csv_path)
    data_frame["ref_path"] = data_frame["filepath_ref"].apply(
        lambda value: data_root / value
    )
    data_frame["deg_path"] = data_frame["filepath_deg"].apply(
        lambda value: data_root / value
    )
    data_frame = data_frame[
        data_frame["ref_path"].apply(Path.exists)
        & data_frame["deg_path"].apply(Path.exists)
    ].copy()
    data_frame = data_frame[
        data_frame["mos"].notna() & (data_frame["mos"] <= mos_max_threshold)
    ].copy()
    data_frame["active_degradation_types"] = data_frame.apply(
        extract_active_degradations, axis=1
    )
    data_frame["num_source_degradation_types"] = data_frame[
        "active_degradation_types"
    ].apply(len)
    if require_active_degradation_types:
        data_frame = data_frame[data_frame["num_source_degradation_types"] > 0].copy()
    return data_frame.reset_index(drop=True)


def prepare_audio_pair(
    row: pd.Series,
    target_sample_rate: int,
    max_duration_seconds: float | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load, resample, and align one REF/DEG audio pair.

    Args:
        row: Source metadata row.
        target_sample_rate: Output sample rate.
        max_duration_seconds: Optional duration cap.

    Returns:
        Aligned REF/DEG pair or None if unusable.
    """

    ref_audio, ref_sample_rate = load_audio_mono(row["ref_path"])
    deg_audio, deg_sample_rate = load_audio_mono(row["deg_path"])
    ref_audio = resample_if_needed(ref_audio, ref_sample_rate, target_sample_rate)
    deg_audio = resample_if_needed(deg_audio, deg_sample_rate, target_sample_rate)
    ref_audio, deg_audio = align_pair(
        ref_audio=ref_audio,
        deg_audio=deg_audio,
        target_sample_rate=target_sample_rate,
        max_duration_seconds=max_duration_seconds,
    )
    if len(ref_audio) < 2 or len(deg_audio) < 2:
        return None
    return ref_audio, deg_audio


def select_accepted_mix(
    ref_audio: np.ndarray,
    deg_audio: np.ndarray,
    target_sample_rate: int,
    config: PlacementConfig,
    rng: random.Random,
) -> tuple[np.ndarray, Segment, float, float] | None:
    """Select one accepted mix for a source pair.

    Args:
        ref_audio: Reference waveform.
        deg_audio: Degraded waveform.
        target_sample_rate: Output sample rate.
        config: Segment placement configuration.
        rng: Random generator.

    Returns:
        Mixed waveform, selected segment, activity threshold, active fraction, or None.
    """

    for _ in range(config.max_row_attempts):
        mixed_audio, segment, threshold, active_fraction = build_mix_one_segment(
            ref_audio=ref_audio,
            deg_audio=deg_audio,
            sample_rate=target_sample_rate,
            config=config,
            rng=rng,
        )
        if active_fraction >= config.output_active_fraction_min:
            return mixed_audio, segment, threshold, active_fraction
    return None


def build_manifest_record(
    index: int,
    src_idx: int,
    pass_idx: int,
    row: pd.Series,
    out_path: Path,
    mixed_audio: np.ndarray,
    selected_segment: Segment,
    selected_threshold: float,
    selected_active_fraction: float,
    target_sample_rate: int,
) -> dict[str, object]:
    """Build one manifest record for a generated mix file.

    Args:
        index: Output index.
        src_idx: Source-row index in filtered dataframe.
        pass_idx: Source pass index.
        row: Source metadata row.
        out_path: Saved output WAV path.
        mixed_audio: Generated mixed waveform.
        selected_segment: Chosen degradation segment.
        selected_threshold: Activity threshold used in selection.
        selected_active_fraction: Active fraction in chosen segment.
        target_sample_rate: Output sample rate.

    Returns:
        Manifest record dictionary.
    """

    duration_seconds = round(len(mixed_audio) / target_sample_rate, 3)
    text_segments = json.dumps([{"start": 0.0, "end": duration_seconds}])
    timeline = timeline_from_segment(
        duration_seconds=duration_seconds, segment=selected_segment
    )
    switch_points = switch_points_from_timeline(timeline)
    return {
        "index": index,
        "filename_ref": row["filename_ref"],
        "filename_deg": row["filename_deg"],
        "mix_filename": out_path.name,
        "mos": float(row["mos"]),
        "duration_seconds": duration_seconds,
        "text_segments": text_segments,
        "mix_deg_segments": serialize_segments([selected_segment]),
        "switch_points": json.dumps(switch_points),
        "mix_timeline": json.dumps(timeline),
        "source_degradation_types": json.dumps(row["active_degradation_types"]),
        "num_source_degradation_types": int(row["num_source_degradation_types"]),
        "activity_threshold": round(float(selected_threshold), 6),
        "segment_active_fraction": round(float(selected_active_fraction), 4),
        "source_row_index": int(src_idx),
        "source_pass": int(pass_idx),
    }


def generate_mixes(
    eligible_rows: pd.DataFrame,
    output_dir: Path,
    target_sample_rate: int,
    max_duration_seconds: float | None,
    total_mix_files: int,
    config: PlacementConfig,
    allow_source_reuse: bool,
    max_source_passes: int,
    rng: random.Random,
) -> pd.DataFrame:
    """Generate mixed files and return the output manifest dataframe.

    Args:
        eligible_rows: Filtered source rows.
        output_dir: Output folder where WAVs are saved.
        target_sample_rate: Output sample rate.
        max_duration_seconds: Optional maximum duration per generated clip.
        total_mix_files: Number of mix files to generate.
        config: Segment placement configuration.
        allow_source_reuse: Whether rows can be reused across source passes.
        max_source_passes: Maximum number of source passes when reuse is enabled.
        rng: Random generator.

    Returns:
        Manifest dataframe.

    Raises:
        ValueError: If generation cannot reach the requested number of files.
    """

    records: list[dict[str, object]] = []
    index_width = max(3, len(str(total_mix_files - 1)))
    pass_limit = max(1, max_source_passes)

    progress = tqdm(total=total_mix_files, desc="Generating mixes", unit="file")
    pass_idx = 0
    while len(records) < total_mix_files and pass_idx < pass_limit:
        pass_idx += 1
        source_indices = list(eligible_rows.index)
        rng.shuffle(source_indices)

        for src_idx in source_indices:
            if len(records) >= total_mix_files:
                break

            row = eligible_rows.loc[src_idx]
            pair = prepare_audio_pair(
                row=row,
                target_sample_rate=target_sample_rate,
                max_duration_seconds=max_duration_seconds,
            )
            if pair is None:
                continue
            ref_audio, deg_audio = pair

            selected = select_accepted_mix(
                ref_audio=ref_audio,
                deg_audio=deg_audio,
                target_sample_rate=target_sample_rate,
                config=config,
                rng=rng,
            )
            if selected is None:
                continue
            (
                selected_mix,
                selected_segment,
                selected_threshold,
                selected_active_fraction,
            ) = selected

            index = len(records)
            stem = Path(row["filename_deg"]).stem
            out_path = output_dir / f"{index:0{index_width}d}_mix_{stem}.wav"
            sf.write(out_path, selected_mix, target_sample_rate)

            records.append(
                build_manifest_record(
                    index=index,
                    src_idx=src_idx,
                    pass_idx=pass_idx,
                    row=row,
                    out_path=out_path,
                    mixed_audio=selected_mix,
                    selected_segment=selected_segment,
                    selected_threshold=selected_threshold,
                    selected_active_fraction=selected_active_fraction,
                    target_sample_rate=target_sample_rate,
                )
            )
            progress.update(1)

        if not allow_source_reuse:
            break

    progress.close()

    if len(records) < total_mix_files:
        raise ValueError(
            f"Generated only {len(records)} files out of requested {total_mix_files}. "
            "Try increasing --mos-max-threshold, lowering --output-active-fraction-min, "
            "increasing --max-row-attempts, or enabling more source passes."
        )

    return pd.DataFrame(records).sort_values("index").reset_index(drop=True)


def main(
    total_mix_files: int = 3000,
    data_root: Path = Path("data/raw/NISQA_Corpus"),
    sim_split: str = "NISQA_TRAIN_SIM",
    output_dir: Path = Path("data/processed/nisqa_sim_mix_lowmos_active_3000"),
    mos_max_threshold: float = 3.0,
    require_active_degradation_types: bool = True,
    target_sample_rate: int = 16000,
    max_duration_seconds: float | None = None,
    seed: int = 42,
    segment_min_seconds: float = 1.0,
    segment_max_ratio: float = 0.40,
    min_active_fraction: float = 0.55,
    activity_std_multiplier: float = 0.65,
    activity_abs_floor: float = 0.006,
    activity_smooth_ms: int = 20,
    segment_attempts: int = 300,
    energy_quantile: float = 0.50,
    energy_top_k_ratio: float = 0.25,
    output_active_fraction_min: float = 0.50,
    max_row_attempts: int = 4,
    allow_source_reuse: bool = True,
    max_source_passes: int = 8,
    overwrite: bool = False,
) -> None:
    """Generate NISQA-SIM low-MOS active-region temporal mixes.

    Args:
        total_mix_files: Number of output mix files to generate.
        data_root: Root path that contains NISQA split folders.
        sim_split: NISQA split name.
        output_dir: Output directory for generated WAV files and manifest.
        mos_max_threshold: Maximum MOS for source selection.
        require_active_degradation_types: Keep only rows with active degradation tags.
        target_sample_rate: Output sample rate.
        max_duration_seconds: Optional max output duration in seconds.
        seed: Random seed.
        segment_min_seconds: Minimum degradation segment duration.
        segment_max_ratio: Maximum degradation segment ratio of full clip length.
        min_active_fraction: Required active-audio coverage for accepted candidate segments.
        activity_std_multiplier: Activity threshold standard-deviation multiplier.
        activity_abs_floor: Minimum absolute activity threshold.
        activity_smooth_ms: Smoothing window for activity envelope.
        segment_attempts: Candidate segment attempts per placement call.
        energy_quantile: Quantile cutoff for energy-based candidate selection.
        energy_top_k_ratio: Fraction of top-energy candidates used as sample pool.
        output_active_fraction_min: Final output active-fraction minimum.
        max_row_attempts: Placement retries per source row.
        allow_source_reuse: Allow reuse of source rows across passes.
        max_source_passes: Max number of shuffled source passes.
        overwrite: Overwrite existing output directory contents.
    """

    rng = random.Random(seed)
    placement_config = PlacementConfig(
        segment_min_seconds=segment_min_seconds,
        segment_max_ratio=segment_max_ratio,
        min_active_fraction=min_active_fraction,
        activity_std_multiplier=activity_std_multiplier,
        activity_abs_floor=activity_abs_floor,
        activity_smooth_ms=activity_smooth_ms,
        segment_attempts=segment_attempts,
        energy_quantile=energy_quantile,
        energy_top_k_ratio=energy_top_k_ratio,
        output_active_fraction_min=output_active_fraction_min,
        max_row_attempts=max_row_attempts,
    )

    prepare_output_dir(output_dir=output_dir, overwrite=overwrite)
    eligible_rows = load_eligible_rows(
        data_root=data_root,
        sim_split=sim_split,
        mos_max_threshold=mos_max_threshold,
        require_active_degradation_types=require_active_degradation_types,
    )
    if not allow_source_reuse and len(eligible_rows) < total_mix_files:
        raise ValueError(
            f"Need at least {total_mix_files} eligible source rows, but found {len(eligible_rows)}. "
            "Enable --allow-source-reuse or increase --mos-max-threshold."
        )

    print(f"Eligible source rows: {len(eligible_rows)}")
    print(f"MOS threshold: <= {mos_max_threshold}")
    print(f"Output dir: {output_dir}")
    print(f"Requested files: {total_mix_files}")

    manifest_df = generate_mixes(
        eligible_rows=eligible_rows,
        output_dir=output_dir,
        target_sample_rate=target_sample_rate,
        max_duration_seconds=max_duration_seconds,
        total_mix_files=total_mix_files,
        config=placement_config,
        allow_source_reuse=allow_source_reuse,
        max_source_passes=max_source_passes,
        rng=rng,
    )
    manifest_path = output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print(f"Wrote {len(manifest_df)} mixed files to {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(
        manifest_df[["mos", "duration_seconds", "segment_active_fraction"]].describe()
    )


if __name__ == "__main__":
    typer.run(main)
