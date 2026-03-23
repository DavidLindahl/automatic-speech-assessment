"""Evaluation routines for zero-shot SALMONN benchmark outputs."""

from __future__ import annotations

from typing import Any

from salmonn_bench.metrics import (
    bleu_score,
    extract_mos,
    extract_winner,
    lcc,
    mean,
    srcc,
)


def evaluate_mos(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate MOS prediction quality.

    Args:
        records: Prediction records containing `mos`, `response`, and `predicted_response`.

    Returns:
        Aggregate metric dictionary and per-sample result rows.
    """
    y_true: list[float] = []
    y_pred: list[float] = []
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    bleu_values: list[float] = []
    rows: list[dict[str, Any]] = []

    for record in records:
        truth = float(record["mos"])
        pred_text = str(record["predicted_response"]).strip()
        pred_mos = extract_mos(pred_text)
        error = abs(truth - pred_mos)
        squared = error**2
        bleu = bleu_score(str(record["response"]), pred_text)

        y_true.append(truth)
        y_pred.append(pred_mos)
        abs_errors.append(error)
        sq_errors.append(squared)
        bleu_values.append(bleu)

        row = dict(record)
        row["predicted_mos"] = pred_mos
        row["mos_error"] = error
        row["bleu"] = bleu
        rows.append(row)

    metrics = {
        "samples": len(rows),
        "mae": mean(abs_errors),
        "mse": mean(sq_errors),
        "lcc": lcc(y_true, y_pred),
        "srcc": srcc(y_true, y_pred),
        "bleu": mean(bleu_values),
    }
    return metrics, rows


def evaluate_ab(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate A/B preference responses.

    Args:
        records: Prediction records containing `winner`, `response`, and `predicted_response`.

    Returns:
        Aggregate metric dictionary and per-sample result rows.
    """
    bleu_values: list[float] = []
    correct = 0
    rows: list[dict[str, Any]] = []

    per_class: dict[str, dict[str, int]] = {
        "A": {"tp": 0, "total": 0},
        "B": {"tp": 0, "total": 0},
        "Tie": {"tp": 0, "total": 0},
    }

    for record in records:
        truth = str(record.get("winner", "Tie"))
        pred_text = str(record["predicted_response"]).strip()
        pred_winner = extract_winner(pred_text)
        is_correct = pred_winner == truth
        if is_correct:
            correct += 1

        if truth in per_class:
            per_class[truth]["total"] += 1
            if is_correct:
                per_class[truth]["tp"] += 1

        bleu = bleu_score(str(record["response"]), pred_text)
        bleu_values.append(bleu)

        row = dict(record)
        row["predicted_winner"] = pred_winner
        row["correct"] = is_correct
        row["bleu"] = bleu
        rows.append(row)

    total = len(rows)
    metrics = {
        "samples": total,
        "accuracy": float(correct / total) if total > 0 else 0.0,
        "per_class": per_class,
        "bleu": mean(bleu_values),
    }
    return metrics, rows
