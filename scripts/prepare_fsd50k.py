from __future__ import annotations

import argparse
from pathlib import Path

from audioforge.data.fsd50k import build_fsd50k_manifests
from audioforge.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build AudioForge FSD50K train/val/test manifests."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/fsd50k"),
        help="Path to extracted FSD50K root.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/manifests/fsd50k"),
        help="Output manifest directory.",
    )

    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Store absolute audio paths in manifests.",
    )

    parser.add_argument(
        "--fail-on-missing-audio",
        action="store_true",
        help="Fail immediately if any audio file referenced by ground truth is missing.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configure_logging(level=args.log_level, json_logs=False)

    logger.info("Building FSD50K manifests")
    logger.info("FSD50K root: %s", args.root)
    logger.info("Output dir: %s", args.out)

    result = build_fsd50k_manifests(
        root=args.root,
        output_dir=args.out,
        absolute_paths=args.absolute_paths,
        fail_on_missing_audio=args.fail_on_missing_audio,
    )

    logger.info("Manifest build complete")
    logger.info("train.csv: %s", result.train_csv)
    logger.info("val.csv: %s", result.val_csv)
    logger.info("test.csv: %s", result.test_csv)
    logger.info("label_map.json: %s", result.label_map_json)
    logger.info("summary.json: %s", result.summary_json)

    for item in result.stats:
        logger.info(
            "%s rows=%s missing_audio=%s duration_hours=%.4f unique_labels=%s",
            item.split,
            item.rows,
            item.missing_audio,
            item.total_duration_hours,
            item.unique_labels,
        )


if __name__ == "__main__":
    main()