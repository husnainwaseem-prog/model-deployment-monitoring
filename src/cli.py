"""CLI: python -m src.cli train | serve"""

import argparse

from .train import train as train_model


def main(argv=None):
    parser = argparse.ArgumentParser(description="Model deployment")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="train the classifier and save the artifact")
    sub.add_parser("serve", help="start the FastAPI service")
    args = parser.parse_args(argv)

    if args.cmd == "train":
        train_model()
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("src.api:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
