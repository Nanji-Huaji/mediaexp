from __future__ import annotations

import argparse
import math
from pathlib import Path

from src.datasets import LocalDatasetManager, SampleCase, load_cases_from_directory
from src.encoder import LZWEncoder, LZSSEncoder, LiteralToken, MatchToken
from src.structure_metrics import analyze_structure


def _lzw_bits(codes: list[int]) -> int:
    if not codes:
        return 0
    width = max(9, math.ceil(math.log2(max(codes) + 1)))
    return len(codes) * width


def _lzss_bits(tokens: list[LiteralToken | MatchToken], encoder: LZSSEncoder) -> int:
    offset_bits = max(1, math.ceil(math.log2(encoder.window_size + 1)))
    length_bits = max(1, math.ceil(math.log2(encoder.lookahead_size + 1)))
    total = 0
    for token in tokens:
        if isinstance(token, LiteralToken):
            total += 1 + 8
        else:
            total += 1 + offset_bits + length_bits
    return total


def summarize_case(
    sample: SampleCase, lzw: LZWEncoder, lzss: LZSSEncoder
) -> dict[str, str | int | float]:
    lzw_codes = lzw.encode(sample.data)
    lzss_tokens = lzss.encode(sample.data)
    structure = analyze_structure(sample.data)

    assert lzw.decode(lzw_codes) == sample.data
    assert lzss.decode(lzss_tokens) == sample.data

    original_bits = len(sample.data) * 8
    lzw_ratio = _lzw_bits(lzw_codes) / original_bits if original_bits else 0.0
    lzss_ratio = _lzss_bits(lzss_tokens, lzss) / original_bits if original_bits else 0.0

    return {
        "category": sample.category,
        "dataset": sample.dataset,
        "name": sample.name,
        "source": str(sample.source) if sample.source is not None else "<memory>",
        "input_bytes": len(sample.data),
        "lzw_units": len(lzw_codes),
        "lzss_units": len(lzss_tokens),
        "structure_score": structure.structure_score,
        "byte_entropy": structure.byte_entropy_bits,
        "window_entropy": structure.window_entropy_bits,
        "lzw_ratio": round(lzw_ratio, 3),
        "lzss_ratio": round(lzss_ratio, 3),
        "detail": sample.detail,
    }


def _print_table(rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        print("No matching files found.")
        return

    headers = [
        "category",
        "dataset",
        "name",
        "input_bytes",
        "structure_score",
        "lzw_ratio",
        "lzss_ratio",
        "detail",
        "source",
    ]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }

    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    separator = "-+-".join("-" * widths[header] for header in headers)
    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))


def _print_summary(rows: list[dict[str, str | int | float]]) -> None:
    by_category: dict[str, list[dict[str, str | int | float]]] = {}
    for row in rows:
        by_category.setdefault(str(row["category"]), []).append(row)

    print()
    print("Average ratios by category:")
    for category, items in sorted(by_category.items()):
        structure_mean = sum(float(item["structure_score"]) for item in items) / len(
            items
        )
        lzw_mean = sum(float(item["lzw_ratio"]) for item in items) / len(items)
        lzss_mean = sum(float(item["lzss_ratio"]) for item in items) / len(items)
        print(
            f"- {category}: structure={structure_mean:.3f}, LZW={lzw_mean:.3f}, LZSS={lzss_mean:.3f}, samples={len(items)}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LZW/LZSS experiments on image and audio directories."
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("assets"),
        help="Managed dataset root containing images/ and audio/",
    )
    parser.add_argument(
        "--image-dir", type=Path, help="Directory containing image samples"
    )
    parser.add_argument(
        "--audio-dir", type=Path, help="Directory containing WAV samples"
    )
    parser.add_argument(
        "--image-datasets",
        nargs="+",
        help="Managed image dataset names under assets/images/",
    )
    parser.add_argument(
        "--audio-datasets",
        nargs="+",
        help="Managed audio dataset names under assets/audio/",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        help="Maximum number of samples to load from each managed dataset",
    )
    return parser


def run_experiment(
    image_dir: Path | None = None,
    audio_dir: Path | None = None,
    asset_root: Path = Path("assets"),
    image_datasets: list[str] | None = None,
    audio_datasets: list[str] | None = None,
    limit_per_dataset: int | None = None,
) -> list[dict[str, str | int | float]]:
    lzw = LZWEncoder()
    lzss = LZSSEncoder()

    cases: list[SampleCase] = []
    if image_dir is not None:
        cases.extend(load_cases_from_directory(image_dir, category="image"))
    if audio_dir is not None:
        cases.extend(load_cases_from_directory(audio_dir, category="audio"))

    manager = LocalDatasetManager(asset_root)
    if image_datasets:
        cases.extend(
            manager.load_cases(
                categories={"image"},
                datasets=set(image_datasets),
                limit_per_dataset=limit_per_dataset,
            )
        )
    if audio_datasets:
        cases.extend(
            manager.load_cases(
                categories={"audio"},
                datasets=set(audio_datasets),
                limit_per_dataset=limit_per_dataset,
            )
        )

    rows = [summarize_case(sample, lzw, lzss) for sample in cases]
    _print_table(rows)
    if rows:
        _print_summary(rows)
    return rows


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if (
        args.image_dir is None
        and args.audio_dir is None
        and not args.image_datasets
        and not args.audio_datasets
    ):
        parser.error(
            "Provide at least one direct directory option or one managed dataset option"
        )

    run_experiment(
        image_dir=args.image_dir,
        audio_dir=args.audio_dir,
        asset_root=args.asset_root,
        image_datasets=args.image_datasets,
        audio_datasets=args.audio_datasets,
        limit_per_dataset=args.limit_per_dataset,
    )


if __name__ == "__main__":
    main()
