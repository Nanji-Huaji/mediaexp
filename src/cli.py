from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.datasets import LocalDatasetManager, import_huggingface_dataset
from src.experiment import main as experiment_main


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset management commands")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("assets"),
        help="Managed dataset root containing images/ and audio/",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("datasets-list", help="List local datasets")
    list_parser.add_argument(
        "--category",
        choices=["image", "audio"],
        help="Only list one category",
    )

    import_parser = subparsers.add_parser(
        "hf-import", help="Import samples from a Hugging Face dataset"
    )
    import_parser.add_argument(
        "--category",
        required=True,
        choices=["image", "audio"],
        help="Dataset category to import",
    )
    import_parser.add_argument(
        "--dataset-id",
        required=True,
        help="Hugging Face dataset id, for example 'mnist' or 'speech_commands'",
    )
    import_parser.add_argument(
        "--dataset-name",
        required=True,
        help="Local dataset name under assets/images or assets/audio",
    )
    import_parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load",
    )
    import_parser.add_argument(
        "--revision",
        help="Optional dataset revision, for example refs/convert/parquet",
    )
    import_parser.add_argument(
        "--config-name",
        help="Optional Hugging Face configuration name",
    )
    import_parser.add_argument(
        "--column",
        help="Column containing image or audio samples; auto-detected when possible",
    )
    import_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of samples to import",
    )

    return parser


def _run_datasets_list(asset_root: Path, category: str | None) -> int:
    manager = LocalDatasetManager(asset_root)
    specs = manager.discover_datasets(category=category)
    if not specs:
        print("No managed datasets found.")
        return 0

    print("category | dataset | file_count | root")
    print("---------+---------+------------+-----")
    for spec in specs:
        print(f"{spec.category} | {spec.name} | {spec.file_count} | {spec.root}")
    return 0


def _run_hf_import(args: argparse.Namespace) -> int:
    imported = import_huggingface_dataset(
        asset_root=args.asset_root,
        category=args.category,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        split=args.split,
        revision=args.revision,
        config_name=args.config_name,
        column=args.column,
        limit=args.limit,
    )
    print(f"Imported {len(imported)} files into {args.asset_root}")
    for path in imported:
        print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0].startswith("-"):
        experiment_main(args)
        return 0

    parser = _build_parser()
    parsed = parser.parse_args(args)

    if parsed.command == "datasets-list":
        return _run_datasets_list(parsed.asset_root, parsed.category)
    if parsed.command == "hf-import":
        return _run_hf_import(parsed)

    parser.error(f"Unknown command: {parsed.command}")
    return 2


__all__ = ["main"]
