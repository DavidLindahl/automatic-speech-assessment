"""Compare variability of chosen/rejected responses between two DPO datasets.

Old: train_dpo_10k.json (sample=False / greedy)
New: train_dpo_10k_plus2epoch_l40s.json (sampled, different temp/top_p)
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    text = path.read_text()
    text_strip = text.lstrip()
    if text_strip.startswith("["):
        return json.loads(text)
    records: list[dict] = []
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        records.append(obj)
        idx = end
    return records


def text_stats(texts: list[str]) -> dict[str, float]:
    char_lens = [len(t) for t in texts]
    word_lens = [len(t.split()) for t in texts]
    unique_ratio = len(set(texts)) / len(texts) if texts else 0.0

    all_words: list[str] = []
    for t in texts:
        all_words.extend(t.lower().split())
    type_token_ratio = len(set(all_words)) / len(all_words) if all_words else 0.0

    first_words = Counter(t.split()[0] if t.split() else "" for t in texts)
    top_first = first_words.most_common(5)

    return {
        "n": len(texts),
        "unique_count": len(set(texts)),
        "unique_ratio": unique_ratio,
        "char_len_mean": statistics.mean(char_lens) if char_lens else 0,
        "char_len_std": statistics.stdev(char_lens) if len(char_lens) > 1 else 0,
        "char_len_min": min(char_lens) if char_lens else 0,
        "char_len_max": max(char_lens) if char_lens else 0,
        "word_len_mean": statistics.mean(word_lens) if word_lens else 0,
        "word_len_std": statistics.stdev(word_lens) if len(word_lens) > 1 else 0,
        "type_token_ratio": type_token_ratio,
        "vocab_size": len(set(all_words)),
        "total_tokens": len(all_words),
        "top_first_words": top_first,
    }


def chosen_equals_rejected_rate(records: list[dict]) -> float:
    same = sum(1 for r in records if r.get("chosen") == r.get("rejected"))
    return same / len(records) if records else 0.0


def chosen_equals_response_rate(records: list[dict]) -> float:
    same = sum(1 for r in records if r.get("chosen") == r.get("response"))
    return same / len(records) if records else 0.0


def report(label: str, records: list[dict]) -> None:
    print(f"\n{'=' * 70}")
    print(f"DATASET: {label}  (n={len(records)})")
    print(f"{'=' * 70}")

    chosen = [r["chosen"] for r in records if "chosen" in r]
    rejected = [r["rejected"] for r in records if "rejected" in r]

    for name, texts in (("CHOSEN", chosen), ("REJECTED", rejected)):
        s = text_stats(texts)
        print(f"\n  {name}")
        print(f"    unique:           {s['unique_count']}/{s['n']}  ({s['unique_ratio']:.4f})")
        print(f"    char len:         mean={s['char_len_mean']:.1f}  std={s['char_len_std']:.1f}  min={s['char_len_min']}  max={s['char_len_max']}")
        print(f"    word len:         mean={s['word_len_mean']:.1f}  std={s['word_len_std']:.1f}")
        print(f"    type-token ratio: {s['type_token_ratio']:.4f}  (vocab={s['vocab_size']}, tokens={s['total_tokens']})")
        print(f"    top first words:  {s['top_first_words']}")

    print(f"\n  CROSS")
    print(f"    chosen == rejected:  {chosen_equals_rejected_rate(records):.4f}")
    print(f"    chosen == response:  {chosen_equals_response_rate(records):.4f}")


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "processed"
    old_path = root / "train_dpo_10k.json"
    new_path = root / "train_dpo_10k_plus2epoch_l40s.json"

    print(f"Loading {old_path.name} ...")
    old = load_records(old_path)
    print(f"Loading {new_path.name} ...")
    new = load_records(new_path)

    report("OLD: train_dpo_10k.json (greedy / sample=False)", old)
    report("NEW: train_dpo_10k_plus2epoch_l40s.json (sampled)", new)

    print(f"\n{'=' * 70}")
    print("DELTA (new vs old)")
    print(f"{'=' * 70}")
    for field in ("chosen", "rejected"):
        old_texts = [r[field] for r in old if field in r]
        new_texts = [r[field] for r in new if field in r]
        old_s = text_stats(old_texts)
        new_s = text_stats(new_texts)
        print(f"\n  {field.upper()}")
        print(f"    unique ratio:     {old_s['unique_ratio']:.4f}  ->  {new_s['unique_ratio']:.4f}   (Δ {new_s['unique_ratio'] - old_s['unique_ratio']:+.4f})")
        print(f"    type-token ratio: {old_s['type_token_ratio']:.4f}  ->  {new_s['type_token_ratio']:.4f}   (Δ {new_s['type_token_ratio'] - old_s['type_token_ratio']:+.4f})")
        print(f"    vocab size:       {old_s['vocab_size']}  ->  {new_s['vocab_size']}   (Δ {new_s['vocab_size'] - old_s['vocab_size']:+d})")
        print(f"    char len std:     {old_s['char_len_std']:.1f}  ->  {new_s['char_len_std']:.1f}   (Δ {new_s['char_len_std'] - old_s['char_len_std']:+.1f})")


if __name__ == "__main__":
    main()
