# imports for parsing to CLI and running commands
import argparse
import subprocess


# functions to run train, fine tune and infer processes
def run_train():
    cmd = [
        "python",
        "train.py",
    ]
    subprocess.run(cmd)


def run_fine_tune():
    cmd = [
        "python",
        "fine_tune.py",
    ]
    subprocess.run(cmd)


def run_infer():
    cmd = [
        "python",
        "weather_forecasting/infer.py",
    ]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Run training, fine-tuning, or inference."
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # train command
    parser_train = subparsers.add_parser("train", help="Train the model")
    parser_train.set_defaults(func=run_train)

    # fine-tune command
    parser_fine_tune = subparsers.add_parser("fine_tune", help="Fine-tune the model")
    parser_fine_tune.set_defaults(func=run_fine_tune)

    # inference command
    parser_infer = subparsers.add_parser("infer", help="Run inference on new data")
    parser_infer.set_defaults(func=run_infer)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
