"""Stage SALMONN-7B runtime assets and write a concrete config.

The DTU /work3 quota can be too tight for Vicuna + Whisper. This helper lets
the LSF job stage model assets into node-local storage, then points the normal
SALMONN benchmark config at those concrete paths.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import hf_hub_download, snapshot_download


SALMONN_REPO = "tsinghua-ee/SALMONN-7B"
PACK_REPO = "fffiloni/SALMONN-7B-PACK"
VICUNA_REPO = "lmsys/vicuna-7b-v1.5"
WHISPER_REPO = "openai/whisper-large-v2"


def _copy_hf_file(repo_id: str, filename: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"OK existing {destination}")
        return
    print(f"Downloading {repo_id}:{filename} -> {destination}")
    source = Path(hf_hub_download(repo_id, filename))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def stage_assets(asset_dir: Path) -> dict[str, Path]:
    asset_dir.mkdir(parents=True, exist_ok=True)

    salmonn_ckpt = asset_dir / "salmonn_7b_v0.pth"
    beats_ckpt = asset_dir / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    vicuna_dir = asset_dir / "vicuna-7b-v1.5"
    whisper_dir = asset_dir / "whisper-large-v2"

    _copy_hf_file(SALMONN_REPO, "salmonn_7b_v0.pth", salmonn_ckpt)
    _copy_hf_file(
        PACK_REPO,
        "beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        beats_ckpt,
    )

    if (vicuna_dir / "config.json").is_file() and any(
        vicuna_dir.glob("pytorch_model-*.bin")
    ):
        print(f"OK existing {vicuna_dir}")
    else:
        print(f"Downloading {VICUNA_REPO} -> {vicuna_dir}")
        snapshot_download(
            VICUNA_REPO,
            local_dir=vicuna_dir,
            ignore_patterns=["*.h5", "*.msgpack", "*.safetensors"],
        )

    if (whisper_dir / "config.json").is_file() and (
        whisper_dir / "model.safetensors"
    ).is_file():
        print(f"OK existing {whisper_dir}")
    else:
        print(f"Downloading {WHISPER_REPO} -> {whisper_dir}")
        snapshot_download(
            WHISPER_REPO,
            local_dir=whisper_dir,
            allow_patterns=[
                "*.json",
                "*.txt",
                "config.json",
                "generation_config.json",
                "merges.txt",
                "model.safetensors",
                "normalizer.json",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ],
        )

    return {
        "llama_path": vicuna_dir.resolve(),
        "whisper_path": whisper_dir.resolve(),
        "beats_path": beats_ckpt.resolve(),
        "ckpt": salmonn_ckpt.resolve(),
    }


def write_runtime_config(
    base_config: Path,
    runtime_config: Path,
    model_paths: dict[str, Path],
) -> None:
    payload: dict[str, Any] = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    model = dict(payload.get("model", {}))
    for key, value in model_paths.items():
        model[key] = str(value)
    payload["model"] = model

    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Wrote runtime config {runtime_config}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/salmonn_zeroshot.yaml"),
    )
    parser.add_argument("--runtime-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_paths = stage_assets(args.asset_dir)
    write_runtime_config(args.base_config, args.runtime_config, model_paths)


if __name__ == "__main__":
    main()
