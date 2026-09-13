"""Download only the six AFVLM-CM image sources required by annotations.

ImageNet-R and Flickr30k are resolved from HaiyangGuo/UCIT. DVQA and
FigureQA are resolved from the live MLLM-CL/FCIT repository tree instead of
assuming archive names. COCO2014 always uses the official COCO HTTP server;
HF_ENDPOINT is never applied to it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = ROOT / "data" / "AFVLM_CM" / "dataset"
PARTITIONS = ROOT / "data" / "AFVLM_CM" / "partitioned"
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip")


def _references() -> set[str]:
    references: set[str] = set()
    for path in PARTITIONS.glob("*_clients/*/*.json"):
        if path.name == "statistics.json":
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            relative = str(row.get("image", "")).replace("\\", "/")
            while relative.startswith("./"):
                relative = relative[2:]
            if relative:
                references.add(relative)
    return references


def _required(prefix: str) -> set[str]:
    marker = prefix.rstrip("/") + "/"
    return {item for item in _references() if item.startswith(marker)}


def _complete(image_root: Path, prefix: str) -> bool:
    required = _required(prefix)
    return bool(required) and all((image_root / item).is_file() for item in required)


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract an archive after rejecting absolute and parent-traversal paths."""
    destination.mkdir(parents=True, exist_ok=True)
    if archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
            if any(
                PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                for name in names
            ):
                raise RuntimeError(f"Unsafe archive member in {archive}")
            handle.extractall(destination)
        return
    with tarfile.open(archive) as handle:
        names = [item.name for item in handle.getmembers()]
        if any(
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts for name in names
        ):
            raise RuntimeError(f"Unsafe archive member in {archive}")
        handle.extractall(destination, filter="data")


def _materialize(staging: Path, image_root: Path, prefix: str) -> None:
    """Copy extracted images to the exact relative paths used by annotations."""
    required = sorted(_required(prefix))
    files = [item for item in staging.rglob("*") if item.is_file()]
    normalized = [(item, item.as_posix().lower()) for item in files]
    by_name: dict[str, list[Path]] = {}
    for item in files:
        by_name.setdefault(item.name.lower(), []).append(item)
    missing: list[str] = []
    ambiguous: list[str] = []
    for relative in required:
        destination = image_root / relative
        if destination.is_file():
            continue
        tail = relative.split("/", 1)[1].lower()
        matches = [item for item, name in normalized if name.endswith("/" + tail)]
        if not matches:
            matches = by_name.get(Path(tail).name.lower(), [])
        if len(matches) == 0:
            missing.append(relative)
            continue
        if len(matches) > 1:
            ambiguous.append(relative)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(matches[0], destination)
    if missing or ambiguous:
        raise RuntimeError(
            f"Could not materialize {prefix}: missing={len(missing)}, "
            f"ambiguous={len(ambiguous)}, examples={(missing + ambiguous)[:10]}"
        )


def _hf_api(endpoint: str | None) -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install project dependencies before downloading HF data") from exc
    return HfApi(endpoint=endpoint)


def _repo_files(repo_id: str, endpoint: str | None) -> list[str]:
    return _hf_api(endpoint).list_repo_files(repo_id=repo_id, repo_type="dataset")


def _snapshot_files(
    repo_id: str,
    files: list[str],
    cache: Path,
    endpoint: str | None,
) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=files,
            cache_dir=cache,
            endpoint=endpoint,
            resume_download=True,
        )
    )


def _archives(files: list[str]) -> list[str]:
    return [item for item in files if item.lower().endswith(ARCHIVE_SUFFIXES)]


def _extract_selected(snapshot: Path, remote_files: list[str], staging: Path) -> None:
    for remote in remote_files:
        archive = snapshot / PurePosixPath(remote)
        if not archive.is_file():
            raise RuntimeError(f"Hugging Face snapshot omitted requested file: {remote}")
        _safe_extract(archive, staging)


def download_ucit(
    kind: str,
    image_root: Path,
    cache: Path,
    endpoint: str | None,
    skip_existing: bool,
) -> None:
    prefix = "ImageNet-R" if kind == "imagenet_r" else "Flickr30k"
    if skip_existing and _complete(image_root, prefix):
        print(f"skip complete {prefix}: {image_root / prefix}")
        return
    files = _repo_files("HaiyangGuo/UCIT", endpoint)
    directory_token = "imagenet-r" if kind == "imagenet_r" else "flickr30k"
    candidates = _archives(
        [item for item in files if directory_token in item.lower().replace("_", "-")]
    )
    if kind == "imagenet_r":
        candidates = [
            item for item in candidates if "imagenetr" in Path(item).name.lower().replace("-", "")
        ]
    else:
        candidates = [
            item
            for item in candidates
            if Path(item).name.lower().split(".", 1)[0] in {"train", "val"}
        ]
    if not candidates:
        raise RuntimeError(
            f"Unable to resolve {prefix} archives from HaiyangGuo/UCIT; "
            f"matching repository entries: "
            f"{[item for item in files if directory_token in item.lower()][:20]}"
        )
    if kind == "imagenet_r" and len(candidates) != 1:
        raise RuntimeError(f"Ambiguous ImageNet-R archives in HaiyangGuo/UCIT: {candidates}")
    snapshot = _snapshot_files("HaiyangGuo/UCIT", candidates, cache, endpoint)
    with tempfile.TemporaryDirectory(dir=cache) as temp:
        staging = Path(temp)
        _extract_selected(snapshot, candidates, staging)
        _materialize(staging, image_root, prefix)


def download_fcit_source(
    dataset_name: str,
    image_root: Path,
    cache: Path,
    endpoint: str | None,
    skip_existing: bool,
) -> None:
    if skip_existing and _complete(image_root, dataset_name):
        print(f"skip complete {dataset_name}: {image_root / dataset_name}")
        return
    files = _repo_files("MLLM-CL/FCIT", endpoint)
    scoped = [
        item
        for item in files
        if "dataset/" in item.lower() and dataset_name.lower() in item.lower()
    ]
    archive_candidates = _archives(scoped)
    if len(archive_candidates) == 1:
        selected = archive_candidates
    elif not archive_candidates:
        direct_images = [
            item
            for item in scoped
            if Path(item).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        if not direct_images:
            raise RuntimeError(
                f"Unable to resolve {dataset_name} files from MLLM-CL/FCIT; "
                f"matching repository entries: {scoped[:20]}"
            )
        selected = direct_images
    else:
        raise RuntimeError(
            f"Unable to resolve a unique {dataset_name} archive from MLLM-CL/FCIT; "
            f"candidates: {archive_candidates}"
        )
    snapshot = _snapshot_files("MLLM-CL/FCIT", selected, cache, endpoint)
    with tempfile.TemporaryDirectory(dir=cache) as temp:
        staging = Path(temp)
        if archive_candidates:
            _extract_selected(snapshot, selected, staging)
        else:
            for remote in selected:
                source = snapshot / PurePosixPath(remote)
                destination = staging / PurePosixPath(remote)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        _materialize(staging, image_root, dataset_name)


def _download_http(
    url: str,
    target: Path,
    resume: bool,
    skip_existing: bool,
    retries: int = 5,
    timeout: int = 60,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and target.is_file():
        print(f"skip existing archive: {target}")
        return
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists() and not resume:
        partial.unlink()
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if resume and partial.exists() else 0
        request = urllib.request.Request(url)
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                append = offset > 0 and response.status == 206
                if offset and not append:
                    offset = 0
                mode = "ab" if append else "wb"
                downloaded = offset
                with partial.open(mode) as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (64 * 1024 * 1024) < len(chunk):
                            print(f"{target.name}: {downloaded / 1024**2:.0f} MiB")
            partial.replace(target)
            return
        except (OSError, urllib.error.URLError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Download failed after {retries} attempts; resumable file: {partial}"
                ) from exc
            time.sleep(min(2**attempt, 30))


def download_coco(
    image_root: Path,
    cache: Path,
    resume: bool,
    skip_existing: bool,
) -> None:
    if skip_existing and _complete(image_root, "COCO2014"):
        print(f"skip complete COCO2014: {image_root / 'COCO2014'}")
        return
    target = image_root / "COCO2014"
    for split in ("train2014", "val2014"):
        if (target / split).is_dir() and skip_existing:
            continue
        archive = cache / f"{split}.zip"
        _download_http(
            f"http://images.cocodataset.org/zips/{split}.zip",
            archive,
            resume,
            skip_existing,
        )
        _safe_extract(archive, target)


def validate(image_root: Path) -> int:
    references = _references()
    missing = [item for item in sorted(references) if not (image_root / item).is_file()]
    print(
        json.dumps(
            {
                "image_root": str(image_root.resolve()),
                "required_unique_images": len(references),
                "missing_images": len(missing),
                "first_missing": missing[:20],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return len(missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--imagenet_r", action="store_true")
    parser.add_argument("--flickr30k", action="store_true")
    parser.add_argument("--dvqa", action="store_true")
    parser.add_argument("--figureqa", action="store_true")
    parser.add_argument("--coco2014", action="store_true", help="Shared by AOKVQA and Grounding")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--hf_endpoint")
    parser.add_argument("--cache_dir", type=Path, default=ROOT / "downloads" / "afvlm_cm")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    args = parser.parse_args()
    image_root = args.output_root.resolve()
    if args.validate_only:
        raise SystemExit(1 if validate(image_root) else 0)
    endpoint = args.hf_endpoint or os.environ.get("HF_ENDPOINT") or None
    selected = {
        "imagenet_r": args.all or args.imagenet_r,
        "flickr30k": args.all or args.flickr30k,
        "dvqa": args.all or args.dvqa,
        "figureqa": args.all or args.figureqa,
        "coco2014": args.all or args.coco2014,
    }
    if not any(selected.values()):
        raise SystemExit("Select --all or at least one dataset flag")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    if selected["imagenet_r"]:
        download_ucit("imagenet_r", image_root, args.cache_dir, endpoint, args.skip_existing)
    if selected["flickr30k"]:
        download_ucit("flickr30k", image_root, args.cache_dir, endpoint, args.skip_existing)
    if selected["dvqa"]:
        download_fcit_source("DVQA", image_root, args.cache_dir, endpoint, args.skip_existing)
    if selected["figureqa"]:
        download_fcit_source("FigureQA", image_root, args.cache_dir, endpoint, args.skip_existing)
    if selected["coco2014"]:
        download_coco(image_root, args.cache_dir, args.resume, args.skip_existing)
    raise SystemExit(1 if validate(image_root) else 0)


if __name__ == "__main__":
    main()
