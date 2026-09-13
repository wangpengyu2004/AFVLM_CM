"""Download the two approved AFVLM-CM checkpoints into atomic local folders."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

MODELS = {
    "llava": ("liuhaotian/llava-v1.5-7b", "llava-v1.5-7b"),
    "clip": ("openai/clip-vit-large-patch14-336", "clip-vit-large-patch14-336"),
}


def download(
    repo_id: str,
    destination: Path,
    endpoint: str | None,
    resume: bool,
    skip_existing: bool,
) -> None:
    from huggingface_hub import snapshot_download

    if (destination / "config.json").is_file() and skip_existing:
        print(f"skip complete checkpoint: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Destination already exists; refusing to overwrite: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() and not resume:
        raise RuntimeError(
            f"Partial download exists: {partial}. Pass --resume or remove it explicitly."
        )
    partial.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=partial,
            local_dir_use_symlinks=False,
            endpoint=endpoint,
            resume_download=resume,
        )
        if not (partial / "config.json").is_file():
            raise RuntimeError(f"Downloaded checkpoint has no config.json: {partial}")
        if destination.exists():
            raise RuntimeError(
                f"Incomplete destination already exists; inspect/remove it: {destination}"
            )
        partial.replace(destination)
    except Exception:
        print(f"download remains resumable in {partial}; no complete directory was created")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf_endpoint")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--llava_only", action="store_true")
    group.add_argument("--clip_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", type=Path, default=Path("pretrained"))
    args = parser.parse_args()
    endpoint = args.hf_endpoint or os.environ.get("HF_ENDPOINT") or None
    selected = ["llava"] if args.llava_only else ["clip"] if args.clip_only else list(MODELS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in selected:
        repo, folder = MODELS[key]
        download(
            repo,
            args.output_dir / folder,
            endpoint,
            args.resume,
            args.skip_existing,
        )


if __name__ == "__main__":
    main()
