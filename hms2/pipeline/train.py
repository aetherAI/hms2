import argparse

from .components.main import train_main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--continue_mode',
        type=bool,
        nargs='?',
        const=True,
        default=False,
    )
    args = parser.parse_args()
    if args.continue_mode:
        print('!!! CONTINUE MODE !!!')

    train_main(args.config, args.continue_mode)
