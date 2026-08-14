import argparse

from system.user_manager import (
    UEBATrainManager,
    UEBADetectionManager,
    UEBAOnlineDetectionManager,
)


def create_parser():
    parser = argparse.ArgumentParser(
        description="UEBA - система детекций аномалий"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # train часть
    train_parser = subparsers.add_parser("train")

    train_parser.add_argument(
        "--train-path",
        default=None,
    )
    train_parser.add_argument(
        "--collect",
        action="store_true",
    )
    train_parser.add_argument(
        "--samples",
        type=int,
        default=1000,
    )
    train_parser.add_argument(
        "--interval",
        type=int,
        default=5,
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    # detect
    detect_parser = subparsers.add_parser("detect")

    detect_parser.add_argument(
        "--data",
        required=True,
    )
    detect_parser.add_argument(
        "--model",
        required=True,
    )
    detect_parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    # online
    online_parser = subparsers.add_parser("online")

    online_parser.add_argument(
        "--model",
        required=True,
    )
    online_parser.add_argument(
        "--interval",
        type=int,
        default=5,
    )

    return parser


def run_train(args):
    manager = UEBATrainManager()

    run_dir = manager.train(
        train_path=args.train_path,
        collect=args.collect,
        samples=args.samples,
        interval=args.interval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print(f"Обучение завершено: {run_dir}")


def run_detect(args):
    manager = UEBADetectionManager()

    result = manager.detect_batch(
        data_path=args.data,
        model_path=args.model,
        batch_size=args.batch_size,
    )

    print("Детекция завершена.")
    print(f"Кол-во объектов: {len(result['errors'])}")
    print(f"Кол-во угроз: {sum(result['alarm'])}")


def run_online(args):
    manager = UEBAOnlineDetectionManager()

    manager.run(
        model_path=args.model,
        interval=args.interval,
    )


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)

    elif args.command == "detect":
        run_detect(args)

    elif args.command == "online":
        run_online(args)


if __name__ == "__main__":
    main()