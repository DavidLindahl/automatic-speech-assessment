"""Diagnostic CLI for validating a DPO preference-pair dataset before training."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import typer

from asa.processed_data import load_processed_records

app = typer.Typer(help="Sanity-check a DPO preference-pair dataset.")

EXPECTED_RECORD_COUNT = 10000
UNIQUE_CHOSEN_FRACTION_THRESHOLD = 0.90
MIN_UNIQUE_REJECTED = 1000
MAX_ABS_MEAN_LENGTH_GAP = 5
MIN_CHOSEN_MOS_COVERAGE = 0.95
MIN_REJECTED_MOS_COVERAGE = 0.80
OUTLIER_REJECTED_LENGTH_FACTOR = 3.0
PREVIEW_CHARS = 140
MOS_PATTERN = re.compile(r"MOS(?:[^0-9]+)(\d+(?:\.\d+)?)")


def word_count(text: str) -> int:
    """Return the number of whitespace-separated tokens in a string.

    Args:
        text: Input string.

    Returns:
        Number of whitespace-delimited tokens.
    """
    return len(text.split())


def percentile(values: list[float], pct: float) -> float:
    """Compute a percentile via linear interpolation on a sorted list.

    Args:
        values: Numeric values, not necessarily sorted.
        pct: Percentile as a float in [0, 100].

    Returns:
        Interpolated percentile value, or 0.0 if the list is empty.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def extract_mos(text: str) -> float | None:
    """Extract the first MOS numeric value mentioned in a string.

    Args:
        text: Candidate caption text.

    Returns:
        The parsed MOS value, or None if the pattern does not match.
    """
    match = MOS_PATTERN.search(text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def truncate(text: str, limit: int = PREVIEW_CHARS) -> str:
    """Return a single-line preview truncated to the given character limit.

    Args:
        text: Source text.
        limit: Maximum preview length in characters.

    Returns:
        Single-line preview string.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."


def summarize_mos_values(values: list[float]) -> str:
    """Render a min/mean/max summary of extracted MOS values.

    Args:
        values: Parsed MOS floats.

    Returns:
        Human-readable summary string.
    """
    if not values:
        return "no values extracted"
    return f"min={min(values):.2f} mean={statistics.mean(values):.2f} max={max(values):.2f}"


def report_record_count(records: list[dict[str, Any]], warnings: list[str]) -> None:
    """Print the record count and warn if it differs from expectation.

    Args:
        records: Parsed DPO records.
        warnings: Mutable list to append soft warnings to.
    """
    total = len(records)
    typer.echo("== Record count ==")
    typer.echo(f"total records: {total}")
    if total != EXPECTED_RECORD_COUNT:
        warnings.append(
            f"record count {total} differs from expected {EXPECTED_RECORD_COUNT}"
        )


def report_uniqueness(
    chosen: list[str],
    rejected: list[str],
    warnings: list[str],
    failures: list[str],
) -> None:
    """Print uniqueness stats and record any hard or soft violations.

    Args:
        chosen: Chosen strings in order.
        rejected: Rejected strings in order.
        warnings: Mutable list of soft warning messages.
        failures: Mutable list of hard-fail messages.
    """
    total = len(chosen)
    unique_chosen = len(set(chosen))
    unique_rejected = len(set(rejected))
    chosen_fraction = unique_chosen / total if total else 0.0
    rejected_fraction = unique_rejected / total if total else 0.0

    typer.echo("== Uniqueness ==")
    typer.echo(f"unique chosen: {unique_chosen}/{total} ({chosen_fraction:.2%})")
    typer.echo(f"unique rejected: {unique_rejected}/{total} ({rejected_fraction:.2%})")

    if chosen_fraction < UNIQUE_CHOSEN_FRACTION_THRESHOLD:
        warnings.append(
            f"unique chosen fraction {chosen_fraction:.2%} below "
            f"{UNIQUE_CHOSEN_FRACTION_THRESHOLD:.0%}"
        )

    if unique_rejected < MIN_UNIQUE_REJECTED:
        failures.append(
            f"unique rejected count {unique_rejected} below minimum {MIN_UNIQUE_REJECTED}"
        )

    counter = Counter(rejected)
    typer.echo("top 3 rejected duplicates:")
    for rank, (text, count) in enumerate(counter.most_common(3), start=1):
        typer.echo(f"  {rank}. count={count} preview={truncate(text)!r}")


def report_length_statistics(
    chosen: list[str],
    rejected: list[str],
    warnings: list[str],
) -> tuple[list[int], list[int], float]:
    """Print length statistics and return word counts plus mean rejected length.

    Args:
        chosen: Chosen strings.
        rejected: Rejected strings.
        warnings: Mutable list to append soft warnings to.

    Returns:
        Tuple of (chosen word counts, rejected word counts, mean rejected word count).
    """
    chosen_words = [word_count(text) for text in chosen]
    rejected_words = [word_count(text) for text in rejected]

    mean_chosen = statistics.mean(chosen_words) if chosen_words else 0.0
    median_chosen = statistics.median(chosen_words) if chosen_words else 0.0
    p90_chosen = percentile([float(v) for v in chosen_words], 90.0)

    mean_rejected = statistics.mean(rejected_words) if rejected_words else 0.0
    median_rejected = statistics.median(rejected_words) if rejected_words else 0.0
    p90_rejected = percentile([float(v) for v in rejected_words], 90.0)

    gaps = [r - c for r, c in zip(rejected_words, chosen_words)]
    mean_gap = statistics.mean(gaps) if gaps else 0.0
    longer_rejected = sum(1 for gap in gaps if gap > 0)
    longer_fraction = longer_rejected / len(gaps) if gaps else 0.0

    typer.echo("== Length statistics (words) ==")
    typer.echo(
        f"chosen   mean={mean_chosen:.2f} median={median_chosen:.2f} p90={p90_chosen:.2f}"
    )
    typer.echo(
        f"rejected mean={mean_rejected:.2f} median={median_rejected:.2f} p90={p90_rejected:.2f}"
    )
    typer.echo(f"mean gap (rejected - chosen): {mean_gap:+.2f} words")
    typer.echo(
        f"rejected strictly longer than chosen: {longer_rejected}/{len(gaps)} "
        f"({longer_fraction:.2%})"
    )

    if abs(mean_gap) > MAX_ABS_MEAN_LENGTH_GAP:
        warnings.append(
            f"|mean length gap| {abs(mean_gap):.2f} words exceeds "
            f"{MAX_ABS_MEAN_LENGTH_GAP}"
        )

    return chosen_words, rejected_words, mean_rejected


def report_mos_coverage(
    chosen: list[str],
    rejected: list[str],
    failures: list[str],
) -> None:
    """Print MOS coverage statistics and record hard failures below thresholds.

    Args:
        chosen: Chosen strings.
        rejected: Rejected strings.
        failures: Mutable list of hard-fail messages.
    """
    total = len(chosen)
    chosen_values = [v for v in (extract_mos(text) for text in chosen) if v is not None]
    rejected_values = [
        v for v in (extract_mos(text) for text in rejected) if v is not None
    ]

    chosen_coverage = len(chosen_values) / total if total else 0.0
    rejected_coverage = len(rejected_values) / total if total else 0.0

    typer.echo("== MOS coverage ==")
    typer.echo(
        f"chosen   matched: {len(chosen_values)}/{total} ({chosen_coverage:.2%}) "
        f"{summarize_mos_values(chosen_values)}"
    )
    typer.echo(
        f"rejected matched: {len(rejected_values)}/{total} ({rejected_coverage:.2%}) "
        f"{summarize_mos_values(rejected_values)}"
    )

    if chosen_coverage < MIN_CHOSEN_MOS_COVERAGE:
        failures.append(
            f"chosen MOS coverage {chosen_coverage:.2%} below "
            f"{MIN_CHOSEN_MOS_COVERAGE:.0%}"
        )
    if rejected_coverage < MIN_REJECTED_MOS_COVERAGE:
        failures.append(
            f"rejected MOS coverage {rejected_coverage:.2%} below "
            f"{MIN_REJECTED_MOS_COVERAGE:.0%}"
        )


def report_degenerate_pairs(
    chosen: list[str],
    rejected: list[str],
    rejected_words: list[int],
    mean_rejected: float,
    warnings: list[str],
    failures: list[str],
) -> None:
    """Print degenerate-pair statistics and record hard or soft violations.

    Args:
        chosen: Chosen strings.
        rejected: Rejected strings.
        rejected_words: Word counts for each rejected string.
        mean_rejected: Mean word count across rejected strings.
        warnings: Mutable list to append soft warnings to.
        failures: Mutable list of hard-fail messages.
    """
    empty_chosen = sum(1 for text in chosen if not text.strip())
    empty_rejected = sum(1 for text in rejected if not text.strip())
    equal_pairs = sum(1 for c, r in zip(chosen, rejected) if c == r)
    outlier_threshold = OUTLIER_REJECTED_LENGTH_FACTOR * mean_rejected
    outliers = sum(1 for wc in rejected_words if wc > outlier_threshold)

    typer.echo("== Degenerate pairs ==")
    typer.echo(f"empty chosen: {empty_chosen}")
    typer.echo(f"empty rejected: {empty_rejected}")
    typer.echo(f"chosen == rejected: {equal_pairs}")
    typer.echo(
        f"rejected length > {OUTLIER_REJECTED_LENGTH_FACTOR:.1f}x mean "
        f"({outlier_threshold:.1f} words): {outliers}"
    )

    if empty_chosen > 0:
        failures.append(f"{empty_chosen} records have empty chosen")
    if empty_rejected > 0:
        failures.append(f"{empty_rejected} records have empty rejected")
    if equal_pairs > 0:
        failures.append(f"{equal_pairs} records have chosen == rejected")
    if outliers > 0:
        warnings.append(
            f"{outliers} rejected strings exceed "
            f"{OUTLIER_REJECTED_LENGTH_FACTOR:.1f}x the mean rejected length"
        )


def report_ground_truth_consistency(
    records: list[dict[str, Any]], warnings: list[str]
) -> None:
    """Check that chosen matches the original response field for every record.

    Args:
        records: Parsed DPO records.
        warnings: Mutable list to append soft warnings to.
    """
    mismatches = sum(
        1
        for record in records
        if record.get("chosen", "") != record.get("response", "")
    )
    typer.echo("== Ground-truth consistency ==")
    typer.echo(f"records where chosen != response: {mismatches}")
    if mismatches > 0:
        warnings.append(
            f"{mismatches} records have chosen != response; DPO pipeline assumes equality"
        )


@app.command()
def check(
    json_path: Path = typer.Argument(..., help="Path to a DPO pair JSON file."),
    strict: bool = typer.Option(True, help="Exit non-zero on hard-fail conditions."),
) -> None:
    """Run the full sanity report on a DPO preference-pair JSON file.

    Args:
        json_path: Path to a DPO pair file (JSON array, JSONL, or object-stream JSON).
        strict: When True, exit with a non-zero status if any hard-fail check fires.
    """
    typer.echo(f"Loading DPO records from {json_path}")
    records = load_processed_records(json_path)

    warnings: list[str] = []
    failures: list[str] = []

    report_record_count(records, warnings)

    if not records:
        failures.append("dataset is empty")
        _finalize(warnings, failures, strict)
        return

    chosen = [str(record.get("chosen", "")) for record in records]
    rejected = [str(record.get("rejected", "")) for record in records]

    report_uniqueness(chosen, rejected, warnings, failures)
    _, rejected_words, mean_rejected = report_length_statistics(
        chosen, rejected, warnings
    )
    report_mos_coverage(chosen, rejected, failures)
    report_degenerate_pairs(
        chosen, rejected, rejected_words, mean_rejected, warnings, failures
    )
    report_ground_truth_consistency(records, warnings)

    _finalize(warnings, failures, strict)


def _finalize(warnings: list[str], failures: list[str], strict: bool) -> None:
    """Print collected warnings and failures, then exit if strict mode requires it.

    Args:
        warnings: Collected soft warning messages.
        failures: Collected hard-fail messages.
        strict: Whether to raise a non-zero exit when failures are present.
    """
    typer.echo("== Summary ==")
    if not warnings and not failures:
        typer.echo("OK: no warnings or failures.")
    for message in warnings:
        typer.echo(f"WARN: {message}")
    for message in failures:
        typer.echo(f"FAIL: {message}")

    if failures and strict:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
