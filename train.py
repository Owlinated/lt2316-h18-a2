# Fix plotting breaking down https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
import matplotlib
matplotlib.use('Agg')

# This is the main training script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.
from keras.callbacks import ModelCheckpoint, History
from keras.utils import plot_model
from matplotlib import pyplot

import mycoco
import pydot # Required for plotting model
from argparse import ArgumentParser
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, SpatialDropout2D
from keras.models import Model
from numpy import array


def create_autoencoder():
    """
    Creates an autoencoder
    :return: Tuple of encoder and autoencoder models
    """
    dropout_rate = 0

    input_img = Input(shape=(200, 200, 3))

    temp_layer = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    temp_layer = MaxPooling2D((5, 5), padding='same')(temp_layer)
    temp_layer = SpatialDropout2D(dropout_rate, data_format='channels_last')(temp_layer)
    temp_layer = Conv2D(8, (3, 3), activation='relu', padding='same')(temp_layer)
    temp_layer = MaxPooling2D((2, 2), padding='same')(temp_layer)
    temp_layer = SpatialDropout2D(dropout_rate, data_format='channels_last')(temp_layer)
    temp_layer = Conv2D(8, (3, 3), activation='relu', padding='same')(temp_layer)

    # (10, 10, 8) encoded
    encoder = MaxPooling2D((2, 2), padding='same', name='encoder')(temp_layer)

    temp_layer = SpatialDropout2D(dropout_rate, data_format='channels_last')(encoder)
    temp_layer = Conv2D(8, (3, 3), activation='relu', padding='same')(temp_layer)
    temp_layer = UpSampling2D((2, 2))(temp_layer)
    temp_layer = SpatialDropout2D(dropout_rate, data_format='channels_last')(temp_layer)
    temp_layer = Conv2D(8, (3, 3), activation='relu', padding='same')(temp_layer)
    temp_layer = UpSampling2D((2, 2))(temp_layer)
    temp_layer = SpatialDropout2D(dropout_rate, data_format='channels_last')(temp_layer)
    temp_layer = Conv2D(16, (3, 3), activation='relu', padding='same')(temp_layer)
    temp_layer = UpSampling2D((5, 5))(temp_layer)
    temp_layer = SpatialDropout2D(dropout_rate, data_format='channels_last')(temp_layer)
    decoder = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(temp_layer)

    autoencoder = Model(input_img, decoder)
    autoencoder.compile(optimizer='adam', loss='mean_absolute_error')

    print(autoencoder.summary())
    plot_model(autoencoder, to_file=args.checkpointdir + '/model.svg')
    return encoder, autoencoder


def autoencoder_generator(iterator, batch_size):
    """
    Turns iterator of tuple(image, category) into generator of tuple(batch(image), batch(image))
    """
    while True:
        batch = []
        for b in range(batch_size):
            sample = next(iterator)
            batch.append(sample[0][0])
        result = array(batch)
        yield result, result


def train_autoencoder(image_iter, image_count):
    (encoder, autoencoder) = create_autoencoder()

    batch_size = 32
    validation_count = 100
    history = History()
    generator = autoencoder_generator(image_iter, batch_size)
    validation_generator = autoencoder_generator(mycoco.iter_all_images(), validation_count)
    autoencoder.fit_generator(
        generator,
        epochs=100,
        steps_per_epoch=image_count / batch_size,
        validation_data=validation_generator,
        validation_steps=validation_count / batch_size,
        callbacks=[
            ModelCheckpoint(args.checkpointdir + '/checkpoint.{epoch:04d}.h5', monitor='loss', mode='auto'),
            history
        ])

    pyplot.figure(figsize=[6, 6])
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('Model Loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['loss', 'val_loss'])
    pyplot.savefig(args.checkpointdir + '/loss.svg', format='svg')

    return encoder, autoencoder


def opt_a():
    """
    Option A - Convolutional image autoencoder
    """
    mycoco.setmode('train')
    image_id_lists = mycoco.query(args.categories)
    image_count = sum(map(lambda list: len(list), image_id_lists))
    image_iter = mycoco.iter_images(image_id_lists, args.categories)
    encoder, autoencoder = train_autoencoder(image_iter, image_count)
    autoencoder.save(args.modelfile)


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
