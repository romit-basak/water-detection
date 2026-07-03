"""
Entry point.

Usage:
    python main.py train                # sanity check on UrbanSARFloods
    python main.py train_three_stage    # Sen1Floods11 pretrain → zero-shot → fine-tune
    python main.py train_pseudo         # pseudo-label training (experimental)
    python main.py evaluate --checkpoint runs/stage3/best.pt
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|train_three_stage|train_pseudo|evaluate]")
        sys.exit(1)

    command = sys.argv.pop(1)

    if command == "train":
        from src.train import train, parse_args
        cfg = parse_args()
        train(cfg)

    elif command == "train_three_stage":
        from src.train_three_stage import main as run, parse_args
        cfg = parse_args()
        run(cfg)

    elif command == "train_pseudo":
        from src.train_pseudo import train, parse_args
        cfg = parse_args()
        train(cfg)

    elif command == "evaluate":
        from src.evaluate import evaluate
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--data_root", default="data/urban_sar_floods")
        p.add_argument("--n_preview", type=int, default=6)
        args = p.parse_args()
        evaluate(args.checkpoint, args.data_root, args.n_preview)

    else:
        print(f"Unknown command: {command}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
