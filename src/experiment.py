from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.datasets import (
    LocalDatasetManager,
    SampleCase,
    load_cases_from_directory,
    load_huggingface_cases,
)
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


def _compute_correlation_and_fit(
    rows: list[dict[str, str | int | float]], ratio_key: str
) -> dict[str, float] | None:
    filtered = [row for row in rows if int(row["input_bytes"]) > 0]
    if len(filtered) < 2:
        return None

    x = np.array([float(row["structure_score"]) for row in filtered], dtype=float)
    y = np.array([float(row[ratio_key]) for row in filtered], dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "correlation": float(correlation),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def _print_correlation_summary(rows: list[dict[str, str | int | float]]) -> None:
    print()
    print("Correlation summary:")
    for ratio_key, label in (("lzw_ratio", "LZW"), ("lzss_ratio", "LZSS")):
        stats = _compute_correlation_and_fit(rows, ratio_key)
        if stats is None:
            print(f"- {label}: insufficient data")
            continue
        print(
            f"- {label}: r={stats['correlation']:.4f}, fit=y={stats['slope']:.4f}x+{stats['intercept']:.4f}"
        )


def _write_csv(rows: list[dict[str, str | int | float]], output_path: Path) -> None:
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _write_correlation_csv(
    rows: list[dict[str, str | int | float]], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "correlation", "slope", "intercept"],
        )
        writer.writeheader()
        for ratio_key, label in (("lzw_ratio", "LZW"), ("lzss_ratio", "LZSS")):
            stats = _compute_correlation_and_fit(rows, ratio_key)
            if stats is None:
                continue
            writer.writerow(
                {
                    "metric": label,
                    "correlation": round(stats["correlation"], 6),
                    "slope": round(stats["slope"], 6),
                    "intercept": round(stats["intercept"], 6),
                }
            )


def _write_structure_plot_pdf(
    rows: list[dict[str, str | int | float]], output_path: Path
) -> None:
    if not rows:
        return

    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Songti SC", "STHeiti", "Arial Unicode MS"):
        if font_name in available_fonts:
            rcParams["font.family"] = font_name
            break
    rcParams["axes.unicode_minus"] = False

    filtered = [row for row in rows if int(row["input_bytes"]) > 0]
    if not filtered:
        return

    groups = [
        ("text", "文本样本"),
        ("image", "图像样本"),
        ("audio", "音频样本"),
        ("all", "全部样本"),
    ]

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (category, title) in zip(axes, groups):
        subset = (
            filtered
            if category == "all"
            else [row for row in filtered if row["category"] == category]
        )
        if not subset:
            ax.set_title(title)
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.grid(True, alpha=0.3)
            continue

        x = np.array([float(row["structure_score"]) for row in subset], dtype=float)
        y_lzw = np.array([float(row["lzw_ratio"]) for row in subset], dtype=float)
        y_lzss = np.array([float(row["lzss_ratio"]) for row in subset], dtype=float)

        ax.scatter(x, y_lzw, label="LZW samples", alpha=0.7, s=26, color="#1f2937")
        ax.scatter(x, y_lzss, label="LZSS samples", alpha=0.7, s=26, color="#6b7280")

        for color, label, ratio_key in (
            ("#2563eb", "LZW fit", "lzw_ratio"),
            ("#dc2626", "LZSS fit", "lzss_ratio"),
        ):
            stats = _compute_correlation_and_fit(subset, ratio_key)
            if stats is None:
                continue
            x_line = np.linspace(0.0, 1.0, 200)
            y_line = stats["slope"] * x_line + stats["intercept"]
            ax.plot(
                x_line,
                y_line,
                color=color,
                linewidth=2.4,
                label=f"{label} (r={stats['correlation']:.3f})",
            )

        lzw_stats = _compute_correlation_and_fit(subset, "lzw_ratio")
        lzss_stats = _compute_correlation_and_fit(subset, "lzss_ratio")
        corr_lines: list[str] = []
        if lzw_stats is not None:
            corr_lines.append(f"LZW r={lzw_stats['correlation']:.3f}")
        if lzss_stats is not None:
            corr_lines.append(f"LZSS r={lzss_stats['correlation']:.3f}")
        if corr_lines:
            ax.text(
                0.03,
                0.97,
                "\n".join(corr_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
            )

        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=True)

    for ax in axes[2:]:
        ax.set_xlabel("Structure Score")
    axes[0].set_ylabel("Compression Ratio")
    axes[2].set_ylabel("Compression Ratio")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def _write_text_repeat_plot_pdf(
    rows: list[dict[str, str | int | float]], output_path: Path
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rcParams

    text_rows = [
        row
        for row in rows
        if row["category"] == "text"
        and int(row["input_bytes"]) > 0
        and float(row.get("text_repeat_frequency", 0.0)) > 0
    ]
    if len(text_rows) < 2:
        return

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Songti SC", "STHeiti", "Arial Unicode MS"):
        if font_name in available_fonts:
            rcParams["font.family"] = font_name
            break
    rcParams["axes.unicode_minus"] = False

    x = np.array([float(row["text_repeat_frequency"]) for row in text_rows], dtype=float)
    y_lzw = np.array([float(row["lzw_ratio"]) for row in text_rows], dtype=float)
    y_lzss = np.array([float(row["lzss_ratio"]) for row in text_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.scatter(x, y_lzw, label="LZW samples", alpha=0.7, s=26, color="#1f2937")
    ax.scatter(x, y_lzss, label="LZSS samples", alpha=0.7, s=26, color="#6b7280")

    for values, color, label in (
        (y_lzw, "#2563eb", "LZW fit"),
        (y_lzss, "#dc2626", "LZSS fit"),
    ):
        if np.std(x) == 0 or np.std(values) == 0:
            continue
        correlation = float(np.corrcoef(x, values)[0, 1])
        slope, intercept = np.polyfit(x, values, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=color, linewidth=2.4, label=f"{label} (r={correlation:.3f})")

    ax.set_xlabel("Text Phrase Repeat Frequency")
    ax.set_ylabel("Compression Ratio")
    ax.set_title("Text Repeat Frequency vs Compression Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


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
        "--text-dir", type=Path, help="Directory containing text samples"
    )
    parser.add_argument(
        "--image-dir", type=Path, help="Directory containing image samples"
    )
    parser.add_argument(
        "--audio-dir", type=Path, help="Directory containing WAV samples"
    )
    parser.add_argument(
        "--text-datasets",
        nargs="+",
        help="Managed text dataset names under assets/text/",
    )
    parser.add_argument(
        "--hf-text-dataset",
        help="Hugging Face text dataset id for direct pipeline use",
    )
    parser.add_argument(
        "--image-datasets",
        nargs="+",
        help="Managed image dataset names under assets/images/",
    )
    parser.add_argument(
        "--hf-image-dataset",
        help="Hugging Face image dataset id for direct pipeline use",
    )
    parser.add_argument(
        "--audio-datasets",
        nargs="+",
        help="Managed audio dataset names under assets/audio/",
    )
    parser.add_argument(
        "--hf-audio-dataset",
        help="Hugging Face audio dataset id for direct pipeline use",
    )
    parser.add_argument(
        "--hf-config-name",
        help="Optional Hugging Face config name for direct pipeline use",
    )
    parser.add_argument(
        "--hf-text-config-name",
        help="Optional Hugging Face config name for direct text pipeline use",
    )
    parser.add_argument(
        "--hf-image-config-name",
        help="Optional Hugging Face config name for direct image pipeline use",
    )
    parser.add_argument(
        "--hf-audio-config-name",
        help="Optional Hugging Face config name for direct audio pipeline use",
    )
    parser.add_argument(
        "--hf-split",
        default="train",
        help="Split used by direct Hugging Face pipeline",
    )
    parser.add_argument(
        "--hf-revision",
        help="Optional revision used by direct Hugging Face pipeline",
    )
    parser.add_argument(
        "--hf-text-column",
        help="Optional text column for direct Hugging Face pipeline",
    )
    parser.add_argument(
        "--hf-image-column",
        help="Optional image column for direct Hugging Face pipeline",
    )
    parser.add_argument(
        "--hf-audio-column",
        help="Optional audio column for direct Hugging Face pipeline",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        help="Maximum number of samples to load from each managed dataset",
    )
    parser.add_argument(
        "--min-text-bytes",
        type=int,
        default=0,
        help="Filter out text samples shorter than this byte length",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Optional CSV output path for experiment rows",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        help="Optional PDF output path for structure-vs-compression plot",
    )
    parser.add_argument(
        "--corr-out",
        type=Path,
        help="Optional CSV output path for correlation and fit statistics",
    )
    parser.add_argument(
        "--text-repeat-plot-out",
        type=Path,
        help="Optional PDF output path for text repeat frequency fitting plot",
    )
    return parser


def run_experiment(
    text_dir: Path | None = None,
    image_dir: Path | None = None,
    audio_dir: Path | None = None,
    asset_root: Path = Path("assets"),
    text_datasets: list[str] | None = None,
    image_datasets: list[str] | None = None,
    audio_datasets: list[str] | None = None,
    hf_text_dataset: str | None = None,
    hf_image_dataset: str | None = None,
    hf_audio_dataset: str | None = None,
    hf_config_name: str | None = None,
    hf_text_config_name: str | None = None,
    hf_image_config_name: str | None = None,
    hf_audio_config_name: str | None = None,
    hf_split: str = "train",
    hf_revision: str | None = None,
    hf_text_column: str | None = None,
    hf_image_column: str | None = None,
    hf_audio_column: str | None = None,
    limit_per_dataset: int | None = None,
    min_text_bytes: int = 0,
    csv_out: Path | None = None,
    plot_out: Path | None = None,
    corr_out: Path | None = None,
    text_repeat_plot_out: Path | None = None,
) -> list[dict[str, str | int | float]]:
    lzw = LZWEncoder()
    lzss = LZSSEncoder()

    cases: list[SampleCase] = []
    if text_dir is not None:
        cases.extend(load_cases_from_directory(text_dir, category="text"))
    if image_dir is not None:
        cases.extend(load_cases_from_directory(image_dir, category="image"))
    if audio_dir is not None:
        cases.extend(load_cases_from_directory(audio_dir, category="audio"))

    manager = LocalDatasetManager(asset_root)
    if text_datasets:
        cases.extend(
            manager.load_cases(
                categories={"text"},
                datasets=set(text_datasets),
                limit_per_dataset=limit_per_dataset,
            )
        )
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
    if hf_text_dataset:
        cases.extend(
            load_huggingface_cases(
                category="text",
                dataset_id=hf_text_dataset,
                split=hf_split,
                revision=hf_revision,
                config_name=hf_text_config_name or hf_config_name,
                column=hf_text_column,
                limit=limit_per_dataset or 10,
                min_text_bytes=min_text_bytes,
            )
        )
    if hf_image_dataset:
        cases.extend(
            load_huggingface_cases(
                category="image",
                dataset_id=hf_image_dataset,
                split=hf_split,
                revision=hf_revision,
                config_name=hf_image_config_name or hf_config_name,
                column=hf_image_column,
                limit=limit_per_dataset or 10,
            )
        )
    if hf_audio_dataset:
        cases.extend(
            load_huggingface_cases(
                category="audio",
                dataset_id=hf_audio_dataset,
                split=hf_split,
                revision=hf_revision,
                config_name=hf_audio_config_name or hf_config_name,
                column=hf_audio_column,
                limit=limit_per_dataset or 10,
            )
        )

    if min_text_bytes > 0:
        cases = [
            sample
            for sample in cases
            if sample.category != "text"
            or sample.dataset == (hf_text_dataset or "")
            or len(sample.data) >= min_text_bytes
        ]

    rows = [
        summarize_case(sample, lzw, lzss)
        for sample in tqdm(cases, desc="Running compression experiment", unit="sample")
    ]
    _print_table(rows)
    if rows:
        _print_summary(rows)
        _print_correlation_summary(rows)
    if csv_out is not None:
        _write_csv(rows, csv_out)
        print(f"\nWrote CSV: {csv_out}")
    if plot_out is not None:
        _write_structure_plot_pdf(rows, plot_out)
        print(f"Wrote plot: {plot_out}")
    if corr_out is not None:
        _write_correlation_csv(rows, corr_out)
        print(f"Wrote correlations: {corr_out}")
    if text_repeat_plot_out is not None:
        _write_text_repeat_plot_pdf(rows, text_repeat_plot_out)
        print(f"Wrote text repeat plot: {text_repeat_plot_out}")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if (
        args.text_dir is None
        and args.image_dir is None
        and args.audio_dir is None
        and not args.text_datasets
        and not args.image_datasets
        and not args.audio_datasets
        and args.hf_text_dataset is None
        and args.hf_image_dataset is None
        and args.hf_audio_dataset is None
    ):
        parser.error(
            "Provide at least one direct directory option or one managed dataset option"
        )

    run_experiment(
        text_dir=args.text_dir,
        image_dir=args.image_dir,
        audio_dir=args.audio_dir,
        asset_root=args.asset_root,
        text_datasets=args.text_datasets,
        image_datasets=args.image_datasets,
        audio_datasets=args.audio_datasets,
        hf_text_dataset=args.hf_text_dataset,
        hf_image_dataset=args.hf_image_dataset,
        hf_audio_dataset=args.hf_audio_dataset,
        hf_config_name=args.hf_config_name,
        hf_text_config_name=args.hf_text_config_name,
        hf_image_config_name=args.hf_image_config_name,
        hf_audio_config_name=args.hf_audio_config_name,
        hf_split=args.hf_split,
        hf_revision=args.hf_revision,
        hf_text_column=args.hf_text_column,
        hf_image_column=args.hf_image_column,
        hf_audio_column=args.hf_audio_column,
        limit_per_dataset=args.limit_per_dataset,
        min_text_bytes=args.min_text_bytes,
        csv_out=args.csv_out,
        plot_out=args.plot_out,
        corr_out=args.corr_out,
        text_repeat_plot_out=args.text_repeat_plot_out,
    )


if __name__ == "__main__":
    main()
