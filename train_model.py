"""
Compatibility wrapper for the canonical main.py train command.

New usage:
    python main.py --mode train --input data/merged/multi_day_train.csv
"""

import argparse
import os
import sys

import main as project_cli


def build_parser():
    parser = argparse.ArgumentParser(
        description="Deprecated wrapper. Prefer: python main.py --mode train"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["split", "single"],
        default="split",
        help="Legacy mode: split uses --split-dir, single uses --input",
    )
    parser.add_argument("--input", type=str, help="Input CSV dataset path for single mode")
    parser.add_argument(
        "--split-dir",
        type=str,
        default="data/processed/split_datasets",
        help="Directory containing train.csv, validation.csv, and test.csv",
    )
    parser.add_argument("--output-dir", type=str, default="models", help="Model output directory")
    parser.add_argument("--model-name", type=str, default="lgbm_model", help="Model base name")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model_path = os.path.join(args.output_dir, f"{args.model_name}.pkl")

    translated = [
        "--mode", "train",
        "--model-path", model_path,
        "--train-ratio", str(args.train_ratio),
        "--valid-ratio", str(args.val_ratio),
    ]

    if args.mode == "single":
        if not args.input:
            print("错误: single模式需要 --input 参数")
            return 1
        translated.extend(["--input", args.input])
    else:
        translated.extend(["--split-dir", args.split_dir])

    print("train_model.py 已兼容保留；建议改用 python main.py --mode train。")
    return project_cli.main(translated)


if __name__ == "__main__":
    sys.exit(main())
