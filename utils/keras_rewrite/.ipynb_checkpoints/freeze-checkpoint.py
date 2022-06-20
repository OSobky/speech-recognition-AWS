import argparse
import tensorflow as tf
import models
import utils
from tensorflow.python.ops import gen_audio_ops as audio_ops

ARGS = None

def main(_ = None):

    utils.assert_exists(output=ARGS.output)

    if ARGS.load_model:
        model = tf.keras.models.load_model(ARGS.load_model)
    else:
        if ARGS.label_count:
            label_count = ARGS.label_count
        elif ARGS.wanted_words:
            # If we specify the words a_1,...,a_n we want the model to be able to distinguish
            # between the labels a_1,...,a_n, silence, unknown. For the input string a_1,...,a_n
            # we can count the number of delimiters ',' and add 1 to get the number of words. We then
            # add 2 to that to include the extra labels silence and unknown.
            label_count = ARGS.wanted_words.count(',') + 3
        else:
            raise Exception("Label count or a list of the wanted words must be provided!")

        utils.assert_exists(model_architecture=ARGS.model_architecture,
                            input_shape=ARGS.input_shape,
                            batch_size=ARGS.batch_size,
                            load_weights=ARGS.load_weights)

        # Create the model and load its weights.
        model = models.create_model(architecture=ARGS.model_architecture.lower(),
                input_shape=utils.parse_list(ARGS.input_shape, type=int, ignore="[]()"), batch_size=ARGS.batch_size,
                label_count=label_count)
        load_status = model.load_weights(ARGS.load_weights)
        print(load_status)

    
    print("Saving to file " + ARGS.output)
    model.save(ARGS.output, save_format=ARGS.save_format)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--output',
    type=str,
    default="./freeze_out/",
    help='Where to save the frozen model.')
    # Import model
    parser.add_argument(
        '--load_model',
        type=str,
        default='',
        help='If specified, load this saved model and extract its weights. Overrides all other arguments except for --h5.'
    )
    # Create new model and load weights
    parser.add_argument(
        '--load_weights',
        type=str,
        default='',
        help='If specified, extract the weights from this saved model and load them into a new model.'
    )
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='',
        help='What model architecture to use')
    parser.add_argument(
        '--wanted_words',
        type=str,
        help='Words that should be decided between. Used to derive label count.',)
    parser.add_argument(
        '--label_count',
        type=int,
        help='Amount of labels to be identified.',)
    parser.add_argument(
        '--input_shape',
        type=str,
        default='(49, 40, 1)',
        help='Input shape of the model',)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size of the model\'s input layer',)

    parser.add_argument(
        '--h5',
        dest='save_format',
        const='h5',
        default='tf',
        help='Save in .hdf5 format.',
        action='store_const')

    ARGS, unparsed = parser.parse_known_args()
    main()