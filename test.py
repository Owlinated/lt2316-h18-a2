# This is the main testing script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.
from matplotlib import pyplot
from numpy import array
from sklearn.decomposition import PCA

import mycoco

# Do not use GPU for testing (it is probably busy training)
from os import environ
print("Disabling GPU")
environ['CUDA_VISIBLE_DEVICES'] = '-1'

from argparse import ArgumentParser
from keras import Model
from keras.models import load_model


def autoencoder_generator(iterator, batch_size):
    """
    Turns iterator of tuple(image, category) into generator of batch(image)
    """
    while True:
        batch = []
        for b in range(batch_size):
            sample = next(iterator)
            batch.append(sample[0][0])
        result = array(batch)
        yield result


def opt_a():
    """
    Option A - Convolutional image autoencoder
    """
    mycoco.setmode('test')

    # Load model
    model: Model = load_model(args.modelfile)
    encoder = Model(inputs=model.input, outputs=model.get_layer("encoder").output)

    # Load image iterator, limit to args.maxinstances per category list
    image_id_lists = mycoco.query(args.categories)
    if args.maxinstances is not None:
        image_id_lists = list(map(lambda list: list[:args.maxinstances], image_id_lists))

    pyplot.figure(figsize=[6, 6])

    for image_id_list in image_id_lists:
        image_count = len(image_id_list)
        image_iter = mycoco.iter_images([image_id_list], args.categories)

        # Create predictions for images
        batch_size = 1
        generator = autoencoder_generator(image_iter, batch_size)
        encoder_prediction = encoder.predict_generator(generator, steps=image_count / batch_size)

        # Reduce dimensionality with PCA
        reshaped_predictions = encoder_prediction.reshape((encoder_prediction.shape[0], -1))
        pca = PCA(n_components=2)
        pca_predictions = pca.fit_transform(reshaped_predictions)

        # Plot values
        pyplot.scatter(pca_predictions[:, 0], pca_predictions[:, 1])

    pyplot.title('Clustering')
    pyplot.legend(args.categories)
    pyplot.savefig('cluster.svg', format='svg')


def opt_b():
    """
    Option B - Multi-task caption predictor/classifier
    """
    mycoco.setmode('test')
    print("Option B not implemented!")


if __name__ == "__main__":
    parser = ArgumentParser("Evaluate a model.")

    # Add your own options as flags HERE as necessary (and some will be necessary!).
    # You shouldn't touch the arguments below.

    parser.add_argument('-P', '--option', type=str,
                        help="Either A or B, based on the version of the assignment you want to run. (REQUIRED)",
                        required=True)
    parser.add_argument('-m', '--maxinstances', type=int,
                        help="The maximum number of instances to be processed per category. (optional)",
                        required=False)
    parser.add_argument('modelfile', type=str, help="model file to evaluate")
    parser.add_argument('categories', metavar='cat', type=str, nargs='+', help='COCO category labels')
    args = parser.parse_args()

    print("Output model in " + args.modelfile)
    print("Maximum instances is " + str(args.maxinstances))

    print("Executing option " + args.option)
    if args.option == 'A':
        opt_a()
    elif args.option == 'B':
        opt_b()
    else:
        print("Option does not exist.")
        exit(0)
