import argparse

from .components.main import visualize_main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
    )
    args = parser.parse_args()
    visualize_main(args.config)
