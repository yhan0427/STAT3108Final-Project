import argparse
from pathlib import Path

from .io import read_json


def main():
    parser = argparse.ArgumentParser(description="Equity volatility research pipeline")
    parser.add_argument("command", choices=["download", "build", "train", "evaluate", "report"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--market-only", action="store_true")
    parser.add_argument("--run", default="main-v1")
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config = read_json(root / "config/study.json")
    universe = read_json(root / "config/universe.json")
    if args.command == "download":
        from .acquire import download
        download(root, config, universe, args.pilot, args.market_only)
    elif args.command == "build":
        from .dataset import build
        build(root, config, universe, args.pilot, args.market_only)
    elif args.command == "train":
        from .experiment import train
        train(root, config, args.run, args.pilot, args.baselines_only)
    elif args.command == "evaluate":
        from .experiment import evaluate
        evaluate(root, args.run)
    else:
        from .report import report
        report(root, args.run)


if __name__ == "__main__":
    main()
