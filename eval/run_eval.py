"""
CLI entrypoint for running the RAG evaluation from a terminal.
"""

import sys
from pathlib import Path
import argparse

# Ensure the project root is on sys.path when this script is invoked from
# a notebook or a non-root working directory.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import ensure_openai_key
from eval.evaluation_runner import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--samples",
        default=str(Path(__file__).resolve().parent.parent / "data" / "eval" / "ragas_samples.json"),
        help="Path to ragas samples json file. Each sample must include an 'id' field for result joining.",
    )
    p.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not prompt for OPENAI_API_KEY if missing in environment",
    )
    args = p.parse_args()

    ensure_openai_key(prompt_if_missing=not args.no_prompt)

    evaluate(args.samples)


if __name__ == "__main__":
    main()