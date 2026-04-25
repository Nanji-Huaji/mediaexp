from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from xml.sax.saxutils import escape

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


def _write_csv(rows: list[dict[str, str | int | float]], output_path: Path) -> None:
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _write_structure_plot(
    rows: list[dict[str, str | int | float]], output_path: Path
) -> None:
    if not rows:
        return

    width = 900
    height = 420
    margin_left = 70
    margin_right = 20
    margin_top = 30
    margin_bottom = 55
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def scale_x(value: float) -> float:
        return margin_left + value * plot_width

    def scale_y(value: float) -> float:
        clamped = max(0.0, min(2.0, value))
        return margin_top + (1.0 - clamped / 2.0) * plot_height

    colors = {"image": "#2563eb", "audio": "#dc2626", "text": "#059669"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:12px} .axis{stroke:#333;stroke-width:1} .grid{stroke:#ddd;stroke-width:1} .label{fill:#111} .legend{font-size:12px}</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
    ]

    for step in range(5):
        x_value = step / 4
        x = scale_x(x_value)
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{height - margin_bottom}"/>'
        )
        parts.append(
            f'<text class="label" x="{x:.1f}" y="{height - margin_bottom + 20}" text-anchor="middle">{x_value:.2f}</text>'
        )

    for step in range(5):
        y_value = step * 0.5
        y = scale_y(y_value)
        parts.append(
            f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="label" x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end">{y_value:.1f}</text>'
        )

    parts.append(
        f'<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}"/>'
    )
    parts.append(
        f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}"/>'
    )
    parts.append(
        f'<text class="label" x="{margin_left + plot_width / 2:.1f}" y="{height - 15}" text-anchor="middle">Structure Score</text>'
    )
    parts.append(
        f'<text class="label" x="20" y="{margin_top + plot_height / 2:.1f}" transform="rotate(-90 20 {margin_top + plot_height / 2:.1f})" text-anchor="middle">Compression Ratio</text>'
    )
    parts.append(
        f'<text class="label" x="{width / 2:.1f}" y="20" text-anchor="middle">Structure vs Compression Ratio</text>'
    )

    legend_y = margin_top + 10
    for index, label in enumerate(["LZW", "LZSS"]):
        cx = width - 180 + index * 80
        color = "#111827" if label == "LZW" else "#6b7280"
        parts.append(f'<circle cx="{cx}" cy="{legend_y}" r="5" fill="{color}"/>')
        parts.append(
            f'<text class="legend" x="{cx + 10}" y="{legend_y + 4}" fill="#111">{label}</text>'
        )

    for row in rows:
        x = scale_x(float(row["structure_score"]))
        y_lzw = scale_y(float(row["lzw_ratio"]))
        y_lzss = scale_y(float(row["lzss_ratio"]))
        category_color = colors.get(str(row["category"]), "#7c3aed")
        label = escape(f"{row['category']}/{row['dataset']}/{row['name']}")

        parts.append(
            f'<circle cx="{x:.1f}" cy="{y_lzw:.1f}" r="5" fill="#111827"><title>{label} LZW={float(row["lzw_ratio"]):.3f}</title></circle>'
        )
        parts.append(
            f'<rect x="{x - 4:.1f}" y="{y_lzss - 4:.1f}" width="8" height="8" fill="#6b7280"><title>{label} LZSS={float(row["lzss_ratio"]):.3f}</title></rect>'
        )
        parts.append(
            f'<line x1="{x:.1f}" y1="{y_lzw:.1f}" x2="{x:.1f}" y2="{y_lzss:.1f}" stroke="{category_color}" stroke-width="1.5" opacity="0.7"/>'
        )

    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


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
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Optional CSV output path for experiment rows",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        help="Optional SVG output path for structure-vs-compression plot",
    )
    return parser


def run_experiment(
    image_dir: Path | None = None,
    audio_dir: Path | None = None,
    asset_root: Path = Path("assets"),
    image_datasets: list[str] | None = None,
    audio_datasets: list[str] | None = None,
    limit_per_dataset: int | None = None,
    csv_out: Path | None = None,
    plot_out: Path | None = None,
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
    if csv_out is not None:
        _write_csv(rows, csv_out)
        print(f"\nWrote CSV: {csv_out}")
    if plot_out is not None:
        _write_structure_plot(rows, plot_out)
        print(f"Wrote plot: {plot_out}")
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

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
        csv_out=args.csv_out,
        plot_out=args.plot_out,
    )


if __name__ == "__main__":
    main()
