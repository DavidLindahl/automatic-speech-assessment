"""Estimate per-component token share of the global and temporal SFT targets.

Segments each target response into ordered spans (glue / timestamp / caption /
MOS), tokenizes each span with the real Qwen2-Audio tokenizer, and averages the
per-component share of total response tokens across the dataset.

This is an estimate: the segmentation is rule-based, so a word assigned to
"caption" vs "glue" depends on the regex boundaries below. The numbers are
meant to show the rough loss-mass split, not an exact accounting.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
GLOBAL_JSON = REPO / "data/processed/sft/train_nisqa_llama_10k.json"
TEMPORAL_JSON = REPO / "data/processed/temporal/train_nisqa_temporal_global_caption.json"
MODEL = "Qwen/Qwen2-Audio-7B"

# Fixed template scaffolding that counts as "glue". These are the only fixed
# phrases the generator inserts around the variable content. Crucially we do
# NOT treat a bare "and" as glue, that conjunction also occurs inside captions
# ("noisy and discontinuous"). Structural connectors that sit right next to a
# timestamp are handled separately in segment(), by position not by word match.
# Anchored with word boundaries so "and" never matches inside "understandable".
GLUE_PHRASES = [
    r"The degradation in the clip is between",
    r",?\s*the overall MOS score is(?:\s+only| about)?",
    r",?\s*[Ww]ith an overall MOS score of",
    r"The overall MOS score is(?:\s+about)?",
]
GLUE_RE = [re.compile(p) for p in GLUE_PHRASES]

# Fixed boilerplate opener ("This/The synthesized speech is/has ..."), up to and
# including the first is/has verb. It is identical on every example and carries
# no learnable signal, so it is glue, not caption. The temporal build strips it
# (splice_caption_from_verb), so counting it as glue makes the *descriptive*
# caption comparable across the two tasks.
OPENER_RE = re.compile(r"^\s*(?:This|The)\b[^.]*?\b(?:is|has)\b", re.IGNORECASE)

TS_RE = re.compile(r"<\|\d+\.\d+\|>")          # timestamp tokens
MOS_RE = re.compile(r"\b\d\.\d\b")              # a MOS value like 1.4
# Connective glue that only counts as glue when adjacent to a timestamp:
# the "and"/"and is" that joins "between <ts> and <ts> and is ...".
CONN_RE = re.compile(r"^\s*and(?:\s+is)?\s*")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def segment(text: str) -> list[tuple[str, str]]:
    """Split a response into ordered (component, span_text) pairs."""
    out_pre: list[tuple[str, str]] = []
    # 0. Peel the fixed boilerplate opener ("This synthesized speech is/has")
    #    as glue. Only the global target carries it; the temporal build already
    #    strips it. Counting it as glue here makes the descriptive caption
    #    comparable across the two tasks.
    om = OPENER_RE.match(text)
    if om:
        out_pre.append(("glue", om.group()))
        text = text[om.end():]

    # 1. Pull out timestamp tokens as their own spans.
    spans: list[tuple[str, str]] = []
    pos = 0
    for m in TS_RE.finditer(text):
        if m.start() > pos:
            spans.append(("_text", text[pos:m.start()]))
        spans.append(("timestamp", m.group()))
        pos = m.end()
    if pos < len(text):
        spans.append(("_text", text[pos:]))

    # 2. Within each plain-text span, peel glue phrases and the MOS number;
    #    everything else is caption. A span that immediately follows a
    #    timestamp may start with a structural connector ("and" / "and is")
    #    that joins the template, only there is "and" glue.
    out: list[tuple[str, str]] = []
    for i, (kind, span) in enumerate(spans):
        if kind == "timestamp":
            out.append((kind, span))
            continue
        prev_is_ts = i > 0 and spans[i - 1][0] == "timestamp"
        if prev_is_ts:
            m = CONN_RE.match(span)
            if m:
                out.append(("glue", m.group()))
                span = span[m.end():]
        out.extend(_segment_text(span))
    return out_pre + out


def _segment_text(span: str) -> list[tuple[str, str]]:
    # Find glue-phrase and MOS-number boundaries, label the gaps as caption.
    marks: list[tuple[int, int, str]] = []
    for rx in GLUE_RE:
        for m in rx.finditer(span):
            marks.append((m.start(), m.end(), "glue"))
    for m in MOS_RE.finditer(span):
        marks.append((m.start(), m.end(), "mos"))
    # Resolve overlaps: prefer earlier-start, then longer span.
    marks.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    chosen: list[tuple[int, int, str]] = []
    last_end = -1
    for s, e, lab in marks:
        if s >= last_end:
            chosen.append((s, e, lab))
            last_end = e

    out: list[tuple[str, str]] = []
    pos = 0
    for s, e, lab in chosen:
        if s > pos:
            out.append(("caption", span[pos:s]))
        out.append((lab, span[s:e]))
        pos = e
    if pos < len(span):
        out.append(("caption", span[pos:]))
    return out


def measure(rows: list[dict], tok) -> dict:
    """Return per-component % share, mean tokens per example, and total tokens."""
    totals = {"glue": 0, "timestamp": 0, "caption": 0, "mos": 0}
    grand = 0
    resp_tokens = 0  # total tokens summed over all responses, for the mean
    for r in rows:
        resp = str(r.get("response", ""))
        for comp, span in segment(resp):
            if not span.strip():
                continue
            n = len(tok(span, add_special_tokens=False)["input_ids"])
            totals[comp] += n
            grand += n
        resp_tokens += len(tok(resp, add_special_tokens=False)["input_ids"])
    n_ex = len(rows)
    return {
        "pct": {k: 100.0 * v / grand for k, v in totals.items()},
        "mean": {k: v / n_ex for k, v in totals.items()},
        "mean_total": resp_tokens / n_ex,
    }


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    glob = load_jsonl(GLOBAL_JSON)
    temp = load_jsonl(TEMPORAL_JSON)
    print(f"global n={len(glob)}  temporal n={len(temp)}")
    g = measure(glob, tok)
    t = measure(temp, tok)
    order = ("caption", "glue", "mos", "timestamp")
    for name, d in (("GLOBAL", g), ("TEMPORAL", t)):
        print(f"\n{name}  (mean total {d['mean_total']:.1f} tokens/example):")
        print(f"  {'component':10s} {'%':>6s} {'mean tok':>9s}")
        for k in order:
            print(f"  {k:10s} {d['pct'][k]:5.1f}% {d['mean'][k]:8.1f}")
    print("\nJSON:", json.dumps({"global": g, "temporal": t}))


if __name__ == "__main__":
    main()
