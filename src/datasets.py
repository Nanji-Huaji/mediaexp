from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import wavfile


IMAGE_EXTENSIONS = {".bmp", ".pgm", ".ppm", ".png", ".jpg", ".jpeg"}
AUDIO_EXTENSIONS = {".wav"}


@dataclass(frozen=True)
class SampleCase:
    category: str
    dataset: str
    name: str
    source: Path | None
    data: bytes
    detail: str


@dataclass(frozen=True)
class DatasetSpec:
    category: str
    name: str
    root: Path
    file_count: int


def _iter_files(directory: Path, extensions: set[str]) -> Iterable[Path]:
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    return (
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in extensions
    )


def _load_image_bytes(path: Path) -> tuple[bytes, str]:
    suffix = path.suffix.lower()

    if suffix in {".bmp", ".pgm", ".ppm"}:
        raw = path.read_bytes()
        return raw, f"file bytes, {len(raw)} bytes"

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading PNG/JPG images requires Pillow. Install it or use BMP/PGM/PPM files."
        ) from exc

    with Image.open(path) as image:
        array = np.asarray(image)
        return array.tobytes(), f"pixels {array.shape}, dtype={array.dtype}"


def _load_audio_bytes(path: Path) -> tuple[bytes, str]:
    sample_rate, samples = wavfile.read(path)
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    frame_count = int(samples.shape[0])
    return (
        np.asarray(samples).tobytes(),
        f"{sample_rate} Hz, {channels} ch, {frame_count} frames, dtype={samples.dtype}",
    )


class LocalDatasetManager:
    def __init__(self, asset_root: Path):
        self.asset_root = asset_root

    def discover_datasets(self, category: str | None = None) -> list[DatasetSpec]:
        categories = [category] if category is not None else ["image", "audio"]
        specs: list[DatasetSpec] = []
        for item in categories:
            specs.extend(self._discover_category(item))
        return specs

    def load_cases(
        self,
        categories: set[str] | None = None,
        datasets: set[str] | None = None,
        limit_per_dataset: int | None = None,
    ) -> list[SampleCase]:
        cases: list[SampleCase] = []
        for spec in self.discover_datasets():
            if categories is not None and spec.category not in categories:
                continue
            if datasets is not None and spec.name not in datasets:
                continue

            paths = list(self._paths_for_spec(spec))
            if limit_per_dataset is not None:
                paths = paths[:limit_per_dataset]
            for path in paths:
                cases.append(self._path_to_case(spec, path))
        return cases

    def _discover_category(self, category: str) -> list[DatasetSpec]:
        category_dir = self._category_dir(category)
        if not category_dir.exists():
            return []

        extensions = self._extensions_for_category(category)
        specs: list[DatasetSpec] = []

        root_files = list(
            path
            for path in sorted(category_dir.iterdir())
            if path.is_file() and path.suffix.lower() in extensions
        )
        if root_files:
            specs.append(
                DatasetSpec(
                    category=category,
                    name="default",
                    root=category_dir,
                    file_count=len(root_files),
                )
            )

        for child in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            file_count = sum(1 for _ in _iter_files(child, extensions))
            if file_count:
                specs.append(
                    DatasetSpec(
                        category=category,
                        name=child.name,
                        root=child,
                        file_count=file_count,
                    )
                )
        return specs

    def _paths_for_spec(self, spec: DatasetSpec) -> Iterable[Path]:
        extensions = self._extensions_for_category(spec.category)
        if spec.name == "default":
            return (
                path
                for path in sorted(spec.root.iterdir())
                if path.is_file() and path.suffix.lower() in extensions
            )
        return _iter_files(spec.root, extensions)

    def _path_to_case(self, spec: DatasetSpec, path: Path) -> SampleCase:
        if spec.category == "image":
            data, detail = _load_image_bytes(path)
        else:
            data, detail = _load_audio_bytes(path)

        return SampleCase(
            category=spec.category,
            dataset=spec.name,
            name=path.stem,
            source=path,
            data=data,
            detail=detail,
        )

    def _category_dir(self, category: str) -> Path:
        if category not in {"image", "audio"}:
            raise ValueError(f"Unsupported category: {category}")
        suffix = "images" if category == "image" else "audio"
        return self.asset_root / suffix

    def _extensions_for_category(self, category: str) -> set[str]:
        if category == "image":
            return IMAGE_EXTENSIONS
        if category == "audio":
            return AUDIO_EXTENSIONS
        raise ValueError(f"Unsupported category: {category}")


def load_cases_from_directory(directory: Path, category: str) -> list[SampleCase]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    extensions = IMAGE_EXTENSIONS if category == "image" else AUDIO_EXTENSIONS
    dataset_name = directory.name
    cases: list[SampleCase] = []
    for path in _iter_files(directory, extensions):
        if category == "image":
            data, detail = _load_image_bytes(path)
        else:
            data, detail = _load_audio_bytes(path)
        cases.append(
            SampleCase(
                category=category,
                dataset=dataset_name,
                name=path.stem,
                source=path,
                data=data,
                detail=detail,
            )
        )
    return cases


__all__ = [
    "AUDIO_EXTENSIONS",
    "DatasetSpec",
    "IMAGE_EXTENSIONS",
    "LocalDatasetManager",
    "SampleCase",
    "load_cases_from_directory",
]
