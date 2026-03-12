import argparse

from .components.main import test_main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
    )
    args = parser.parse_args()
    test_main(args.config)
