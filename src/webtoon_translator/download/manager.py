"""Model download manager.

Downloads pinned model snapshots from the Hugging Face Hub into the app data
directory, file by file so we can report progress to the GUI.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ..paths import models_dir
from .manifest import MODELS, ModelSpec

log = logging.getLogger(__name__)

# progress callback: (model_key, filename, files_done, files_total)
ProgressCb = Callable[[str, str, int, int], None]


def model_path(key: str) -> Path:
    return models_dir() / key


def _required_files(spec: ModelSpec) -> list[str]:
    from huggingface_hub import list_repo_files
    from huggingface_hub.utils import filter_repo_objects

    files = list_repo_files(spec.repo_id, revision=spec.revision)
    if spec.allow_patterns:
        files = list(filter_repo_objects(files, allow_patterns=list(spec.allow_patterns)))
    return files


def is_downloaded(key: str) -> bool:
    marker = model_path(key) / ".complete"
    return marker.exists()


def missing_models() -> list[ModelSpec]:
    return [spec for key, spec in MODELS.items() if not is_downloaded(key)]


def download_model(
    spec: ModelSpec,
    progress_cb: ProgressCb | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download

    target = model_path(spec.key)
    target.mkdir(parents=True, exist_ok=True)
    files = _required_files(spec)
    for i, filename in enumerate(files):
        if cancel_check and cancel_check():
            raise InterruptedError("download cancelled")
        if progress_cb:
            progress_cb(spec.key, filename, i, len(files))
        hf_hub_download(
            spec.repo_id,
            filename,
            revision=spec.revision,
            local_dir=target,
        )
    if progress_cb:
        progress_cb(spec.key, "", len(files), len(files))
    (target / ".complete").write_text(spec.revision)
    log.info("downloaded %s -> %s", spec.repo_id, target)
    return target


def ensure_all(
    progress_cb: ProgressCb | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    for spec in missing_models():
        download_model(spec, progress_cb, cancel_check)
