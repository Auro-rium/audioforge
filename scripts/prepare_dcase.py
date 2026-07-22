#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from audioforge.data.dcase import build_dcase_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DCASE audio manifest.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/manifests/dcase2024/all.csv"))
    args = parser.parse_args()
    stats = build_dcase_manifest(args.root, args.out)
    print(stats)


if __name__ == "__main__":
    main()
