from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("algorithm", help="Name of algorithm to train",
                    choices=["DescriptionExtractor", "Sanitizer"], type=str)
parser.add_argument("-m", "--model", help="Model path", default="./models/", type=str)
parser.add_argument("-o", "--override", help="If model should be overridden", default=0, type=int)
parser.add_argument("-e", "--epochs", help="How many epochs to train", default=50, type=int)
parser.add_argument("-p", "--print", help="How often per iteration to print update", default=40, type=int)
parser.add_argument("-d", "--dataset", help="What dataset to train", type=str)
parser.add_argument("-v", "--visualize", help="If training should be visualized with matplotlib", default=0, type=int)
parser.add_argument("-t", "--tensorboard", help="If training should be logged with tensorboard", default=0, type=int)
parser.add_argument("--lmdb", help="Whether to use LMDB database or not", default=1, type=int)


def main():
    args = parser.parse_args()
    for arg in vars(args):
        print("\t", arg, getattr(args, arg))
    print("\n")

    modelPath = args.model
    if os.path.isdir(modelPath):
        modelPath = os.path.join(modelPath, args.algorithm+".pth")
    alreadyExists = os.path.exists(modelPath)

    if args.algorithm == "DescriptionExtractor":
        pass  # TODO

    elif args.algorithm == "Sanitizer":
        from DenseSense.algorithms.Sanitizer import Sanitizer
        sanitizer = Sanitizer()
        if alreadyExists and not args.override:
            print("Will keep working on existing model")
            sanitizer.loadModel(modelPath)
        sanitizer.saveModel(modelPath)

        dataset = "val2017"
        if args.dataset is not None:
            dataset = args.dataset

        sanitizer.train(epochs=args.epochs, dataset=dataset,
                        useDatabase=args.lmdb, printUpdateEvery=args.print,
                        visualize=args.visualize, tensorboard=args.tensorboard)


if __name__ == '__main__':
    main()
