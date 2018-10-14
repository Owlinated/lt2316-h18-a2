# This is the main training script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.

from argparse import ArgumentParser
import mycoco


def opt_a():
    """
    Option A - Convolutional image autoencoder
    """
    mycoco.setmode('train')
    print("Option A not implemented!")


def opt_b():
    """
    Option B - Multi-task caption predictor/classifier
    """
    mycoco.setmode('train')
    print("Option B not implemented!")


if __name__ == "__main__":
    parser = ArgumentParser("Train a model.")

    # Add your own options as flags HERE as necessary (and some will be necessary!).
    # You shouldn't touch the arguments below.

    parser.add_argument('-P', '--option', type=str,
                        help="Either A or B, based on the version of the assignment you want to run. (REQUIRED)",
                        required=True)
    parser.add_argument('-m', '--maxinstances', type=int,
                        help="The maximum number of instances to be processed per category. (optional)",
                        required=False)
    parser.add_argument('checkpointdir', type=str,
                        help="directory for storing checkpointed models and other metadata (recommended to create a directory under /scratch/)")
    parser.add_argument('modelfile', type=str, help="output model file")
    parser.add_argument('categories', metavar='cat', type=str, nargs='+',
                        help='two or more COCO category labels')
    args = parser.parse_args()

    print("Output model in " + args.modelfile)
    print("Working directory at " + args.checkpointdir)
    print("Maximum instances is " + str(args.maxinstances))

    if len(args.categories) < 2:
        print("Too few categories (<2).")
        exit(1)

    print("The queried COCO categories are:")
    for c in args.categories:
        print("\t" + c)

    print("Executing option " + args.option)
    if args.option == 'A':
        opt_a()
    elif args.option == 'B':
        opt_b()
    else:
        print("Option does not exist.")
        exit(1)
