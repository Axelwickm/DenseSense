from argparse import ArgumentParser
import os
import shutil

parser = ArgumentParser()
trainable = ["DescriptionExtractor", "Sanitizer", "ActionClassifier"]
parser.add_argument("algorithm", help="Name of algorithm to train",
                    choices=trainable, type=str)
parser.add_argument("-m", "--model", help="Model path", default="./models/", type=str)
parser.add_argument("-o", "--override", help="If model should be overridden", default=0, type=int)
parser.add_argument("-e", "--epochs", help="How many epochs to train", default=50, type=int)
parser.add_argument("-lr", "--learningRate", help="How many epochs to train", default=0.005, type=float)
parser.add_argument("-p", "--print", help="How often per iteration to print update", default=40, type=int)
parser.add_argument("-d", "--dataset", help="What dataset to train", type=str)
parser.add_argument("-v", "--visualize", help="If training should be visualized with matplotlib", default=0, type=int)
parser.add_argument("-t", "--tensorboard", help="If training should be logged with tensorboard", default="0", type=str)
parser.add_argument("--lmdb", help="Whether to use LMDB database or not", default=1, type=int)


def main():
    # Print args
    args = parser.parse_args()
    for arg in vars(args):
        print("\t", arg, getattr(args, arg))
    print("\n")

    # Determine model path
    modelPath = args.model
    if os.path.isdir(modelPath):
        modelPath = os.path.join(modelPath, args.algorithm+".pth")
    alreadyExists = os.path.exists(modelPath)

    # Determine tensorboard path
    try:
        tb = int(args.tensorboard)
        tb = True if 0 < tb else False
    except ValueError:
        tb = args.tensorboard

        # Potentially delete old tensorboard
        if os.path.isdir("./data/tensorboard/"+tb):
            print("Deleting old tensorboard: "+tb)
            shutil.rmtree("./data/tensorboard/"+tb)

    if args.algorithm == "DescriptionExtractor":
        from DenseSense.algorithms.DescriptionExtractor import DescriptionExtractor
        descriptionExtractor = DescriptionExtractor()
        if alreadyExists and not args.override:
            print("Will keep working on existing model")
            descriptionExtractor.loadModel(modelPath)
        descriptionExtractor.saveModel(modelPath)

        dataset = "val"
        if args.dataset is not None:
            dataset = args.dataset

        descriptionExtractor.train(epochs=args.epochs, dataset=dataset, learningRate=args.learningRate,
                        useDatabase=args.lmdb, printUpdateEvery=args.print,
                        visualize=args.visualize, tensorboard=tb)

    elif args.algorithm == "Sanitizer":
        from DenseSense.algorithms.Sanitizer import Sanitizer
        sanitizer = Sanitizer()
        if alreadyExists and not args.override:
            print("Will keep working on existing model")
            sanitizer.load_model(modelPath)
        sanitizer.save_model(modelPath)

        dataset = "val2017"
        if args.dataset is not None:
            dataset = args.dataset

        sanitizer.train(epochs=args.epochs, dataset=dataset, learning_rate=args.learningRate,
                        use_database=args.lmdb, print_update_every=args.print,
                        visualize=args.visualize, tensorboard=tb)

    elif args.algorithm == "ActionClassifier":
        from DenseSense.algorithms.ActionClassifier import ActionClassifier
        ac = ActionClassifier()
        if alreadyExists and not args.override:
            print("Will keep working on existing model")
            ac.loadModel(modelPath)
        ac.saveModel(modelPath)

        dataset = "val2017"
        if args.dataset is not None:
            dataset = args.dataset

        ac.trainAutoEncoder(epochs=args.epochs, dataset=dataset, learningRate=args.learningRate,
                            useLMDB=args.lmdb, printUpdateEvery=args.print,
                            visualize=args.visualize, tensorboard=tb)


if __name__ == '__main__':
    main()
