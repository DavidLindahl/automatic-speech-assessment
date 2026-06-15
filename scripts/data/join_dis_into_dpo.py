"""Join NISQA discontinuity (`dis`) onto an existing global DPO preference file.

For the discontinuity ablation (\autoref{sec:results-ablation-dis}) we want the
cleanest possible 4-vs-5-dim comparison: the SAME chosen/rejected pairs, only the
text reference's metadata changes. The existing global preference file
(``train_dpo_full_sft.json``) carries ``mos/noi/col/loud`` but not ``dis``. This
script stamps ``dis`` onto each record by joining the raw NISQA-SIM CSV on the
degraded-clip basename, and writes a new file. Nothing else changes: chosen,
rejected, query, audios are copied verbatim.

The 5-dim ALLD run then reads this file with ``--use-discontinuity``, so the only
difference from the 4-dim baseline is that the reference prompt's Input line shows
``{mos, noi, col, dis, loud}`` instead of ``{mos, noi, col, loud}``.

Usage:
  python scripts/data/join_dis_into_dpo.py \
    --in-json  data/processed/dpo/train_dpo_full_sft.json \
    --csv      data/raw/NISQA_Corpus/NISQA_TRAIN_SIM/NISQA_TRAIN_SIM_file.csv \
    --out-json data/processed/dpo/train_dpo_full_sft_dis.json
"""

import csv
import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Join NISQA `dis` onto an existing DPO preference file.")


def _build_dis_index(csv_path: Path) -> dict[str, float]:
    """Map degraded-clip basename -> NISQA discontinuity (`dis`) score."""
    index: dict[str, float] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            deg = row.get("filename_deg") or row.get("deg") or row.get("filepath_deg")
            dis = row.get("dis")
            if not deg or dis in (None, ""):
                continue
            try:
                index[Path(deg).name] = float(dis)
            except ValueError:
                continue
    return index


def _deg_basename(record: dict) -> Optional[str]:
    """Pull the degraded-clip basename from a record's `audios` field."""
    audios = record.get("audios")
    if isinstance(audios, list) and audios:
        return Path(str(audios[0])).name
    if isinstance(audios, str) and audios:
        return Path(audios).name
    return None


@app.command()
def main(
    in_json: Path = typer.Option(..., "--in-json", help="Existing DPO preference file (JSON array)."),
    csv_path: Path = typer.Option(..., "--csv", help="NISQA_TRAIN_SIM_file.csv with the `dis` column."),
    out_json: Path = typer.Option(..., "--out-json", help="Output file with `dis` joined onto each record."),
) -> None:
    records = json.loads(in_json.read_text())
    if not isinstance(records, list):
        raise typer.BadParameter("Input must be a JSON array of records.")

    dis_index = _build_dis_index(csv_path)
    typer.echo(f"dis index: {len(dis_index)} clips from {csv_path.name}")

    matched = 0
    missing = 0
    for rec in records:
        name = _deg_basename(rec)
        dis = dis_index.get(name) if name else None
        if dis is None:
            missing += 1
            continue
        rec["dis"] = dis
        matched += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records))
    typer.echo(
        f"Wrote {len(records)} records to {out_json} "
        f"({matched} with dis, {missing} unmatched)."
    )
    if missing:
        typer.echo(
            f"WARNING: {missing} records had no dis match; "
            f"they keep their original 4-dim fields and will error under "
            f"--use-discontinuity. Investigate before training if nonzero."
        )


if __name__ == "__main__":
    app()
