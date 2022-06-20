
import os.path
import models
import utils
import preprocessing as pp
import argparse
import math

import tensorflow as tf
import tensorflow.keras as keras


import mlflow
import mlflow.tensorflow


from dotenv import load_dotenv
from functools import reduce

import matplotlib.pyplot as plt

# TODO:

# '--eval_freq',
# '--optimizer', <- gradient_descent vs momentum? add adam?

# TODO verbosity
# the verbosity should be revised, since most keras operations differentiate between two levels of verbosity

# Possible hyperparameters:
#  optimizer params
#  learning_rate
#  learning_rate_schedule
#  batch_size
#  preprocessing params?

# TODO log summaries with tensorboard





load_dotenv(verbose=True)

def prepare_dataset_for_training(dataset, current_size, required_size, shuffle_buffer_size=100, seed=59185):
    return dataset.shuffle(shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True)\
                .batch(ARGS.batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)\
                .cache()\
                .repeat(math.ceil(required_size/current_size) if required_size > 0 else 1)\
                .prefetch(tf.data.AUTOTUNE)

def find_input_shape(example_wav_path):
    return pp.wav_to_spectrogram(tf.io.read_file(example_wav_path),
                                ARGS.sample_rate,
                                ARGS.clip_duration_ms,
                                ARGS.window_size_ms,
                                ARGS.window_stride_ms,
                                ARGS.feature_bin_count,
                                preprocessing_mode=ARGS.preprocess).shape


def main():

    # MLflow setup
    mlflow.tensorflow.autolog(every_n_iter=1)

    utils.download_data('https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
                    ARGS.data_dir,
                    extract=True
                    )

    # Preprocessing
    pp.set_seed(ARGS.seed)

    data_index, label_to_index, all_labels = pp.create_data_index(ARGS.data_dir, utils.parse_list(ARGS.wanted_words),
                                        val_pc=ARGS.validation_percentage, test_pc=ARGS.testing_percentage,
                                        silence_pc=ARGS.silence_percentage, unknown_pc=ARGS.unknown_percentage, ignore=ARGS.bg_noise_subdir, verbose=ARGS.verbosity)

    training_steps = utils.parse_list(ARGS.training_steps, int)
    required_size = sum(training_steps) * ARGS.epochs * (ARGS.batch_size + 1)
    bg_vol_interval = utils.parse_list(ARGS.bg_volume_interval, float)
    
    datasets = pp.construct_datasets(data_index, label_to_index, ARGS.sample_rate, ARGS.clip_duration_ms,
                                window_size_ms=ARGS.window_size_ms, window_stride_ms=ARGS.window_stride_ms, feature_bin_count=ARGS.feature_bin_count, preprocessing_mode=ARGS.preprocess,
                                bg_noise_dir=os.path.join(ARGS.data_dir, ARGS.bg_noise_subdir), bg_noise_freq=ARGS.bg_noise_freq, min_bg_noise=bg_vol_interval[0], max_bg_noise=bg_vol_interval[1],
                                time_shift_ms=ARGS.time_shift_ms, verbose=ARGS.verbosity)

    datasets = utils.map_dict(lambda p,d: prepare_dataset_for_training(d, len(data_index[p]), required_size if p=='training' else -1, seed = ARGS.seed), datasets)
    
    # Convert training_steps presented as (s_1, s_2, ..., s_n) to (s_1, s_1 + s_2, ..., s_1 + ... + s_n)
    training_steps = reduce(lambda x,y: x + [(x[-1] if x else 0) + y], training_steps, [])

    # Make sure len(learning_rates) = len(training_steps) + 1, crop to the length or fill up by repeating the last element
    learning_rates = utils.parse_list(ARGS.learning_rate, float)[:len(training_steps)+1]
    learning_rates += [learning_rates[-1]] * (len(training_steps) - len(learning_rates) + 1)

    input_shape = utils.parse_list(ARGS.input_shape, type=int) if ARGS.input_shape else find_input_shape(data_index['training'][0][0])
    
    print(input_shape)

    # Create the model that will be used in training
    if ARGS.start_checkpoint:
        if not utils.path_exists(ARGS.start_checkpoint):
            raise Exception("Could not load checkpoint. File '" + ARGS.start_checkpoint + "' not found.")
        model = keras.models.load_model(ARGS.start_checkpoint)
    else:
        model = models.create_model(architecture=ARGS.model_architecture.lower(),
            input_shape=input_shape, batch_size=ARGS.batch_size,
            # If we specify the words a_1,...,a_n we want the model to be able to distinguish
            # between the labels a_1,...,a_n, silence, unknown. For the input string a_1,...,a_n
            # we can count the number of delimiters ',' and add 1 to get the number of words. We then
            # add 2 to that to include the extra labels silence and unknown.
            label_count=ARGS.wanted_words.count(',') + 3)

        # Learning rate scheduler
        lr_scheduler = keras.optimizers.schedules.PiecewiseConstantDecay(training_steps, learning_rates)

        # Prepare the model for training
        model.compile(  optimizer=keras.optimizers.SGD(learning_rate=lr_scheduler, clipnorm=1.),
                        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        metrics=['sparse_categorical_accuracy'],
                    )
    
    # Print model summary
    if ARGS.verbosity:
        print("\nModel creation complete.")
        model.summary()

    train = lambda run_id : model.fit(datasets['training'],
                        validation_data=datasets['validation'],
                        epochs=ARGS.epochs,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='loss', verbose=1, patience=2),
                        keras.callbacks.ModelCheckpoint(filepath=os.path.join(ARGS.train_dir, 'run_' + run_id, 'ckpt', 'weights-{epoch:02d}-{sparse_categorical_accuracy:.4f}.' + ARGS.save_format),
                                        monitor='sparse_categorical_accuracy',
                                        mode='max',
                                        save_best_only=True,
                                        verbose=1,
                                        save_freq="epoch" if ARGS.save_freq <= 0 else ARGS.save_freq,
                                        ),],
                        steps_per_epoch=training_steps[-1]
                        )
    
    if not ARGS.no_mlflow:
        # Training and evaluation
        with mlflow.start_run():
            run_id = mlflow.active_run().info.run_id
            history = train(run_id)
    else:
        history = train('local')

    # Display metrics on a plot
    metrics = history.history
    print(metrics)

    plt.plot(history.epoch, metrics['loss'], metrics['sparse_categorical_accuracy'])
    plt.legend(['loss', 'sparse_categorical_accuracy'])
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Important paths
    parser.add_argument(
        '--data_url',
        type=str,
        # pylint: disable=line-too-long
        default='https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
        # pylint: enable=line-too-long
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/speech_dataset/',
        help="""\
        Where to download the speech training data to.
        """)
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/tmp/microspeech_logs',
        help='Where to save summary logs for TensorBoard.')
    parser.add_argument(
        '--train_dir',
        type=str,
        default='/tmp/microspeech_train',
        help='Directory to write event logs and checkpoints.')


    # Dataset settings
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)',)
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
        How much of the training data should be silence.
        """)
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
        How much of the training data should be unknown words.
        """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    
    # Preprocessing settings
    parser.add_argument(
        '--bg_noise_subdir',
        type=str,
        default='_background_noise_',
        help="""\
            Name or relative path of the subdirectory containing background noise files. The subdirectory
            should be located in 'data_dir'. If a relative path is given, it should be relative to the path of 'data_dir'
        """)
    parser.add_argument(
        '--bg_volume_interval',
        type=str,
        default="0.,0.1",
        help="""\
        How loud the background noise should be. Expects two floating point numbers in the interval [0, 1] separated by a comma (eg. "0.25, 0.75"). The volume level of the background noise added will be in this range.
        """)
    parser.add_argument(
        '--bg_noise_freq',
        type=float,
        default=0.8,
        help="""\
        How many of the training samples have background noise mixed in.
        """)
    parser.add_argument(
        '--time_shift_ms',
        type=float,
        default=100.0,
        help="""\
        Range to randomly shift the training audio by in time.
        """)
    parser.add_argument(
        '--preprocess',
        type=str,
        default='micro',
        help='Spectrogram processing mode. Can be "mfcc", "average", or "micro"')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',)
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs',)
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is.',)
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How far to move in time between spectrogram timeslices.',
    )
    parser.add_argument(
        '--feature_bin_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint',
    )


    # Model settings
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='tiny_conv',
        help="""\
        What model architecture to use. Available options are 'conv', 'tiny_conv', 'tiny_embedding_conv',
        'single_fc', 'low_latency_conv', or 'low_latency_svdf'.
        """)
    parser.add_argument(
        '--input_shape',
        type=str,
        default='',
        help='If specified use this input shape for the model. Otherwise deduce input shape from preprocessed data.',)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once',)
    
    # Training settings
    parser.add_argument(
        '--training_steps',
        type=str,
        default='12000,3000',
        help="""\
        How many training steps there should be per epoch. Expects a list of ints separated by commas.
        Specify the learning rates for each interval using the option --learning_rate
        """,)
    parser.add_argument(
        '--learning_rate',
        type=str,
        default='0.0001,0.00001',
        help="""\
        How large the learning rate should be when training. Expects a list of floats separated 
        by commas corresponding to the training_steps specified.
        """)
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='How many epochs to train for.',)
    parser.add_argument(
        '--save_freq',
        type=int,
        default=-1,
        help="Steps after which a model checkpoint is made. Defaults to -1, which saves a model checkpoint at the end of every epoch.")
    parser.add_argument(
        '--h5',
        const='tf',
        default='h5',
        dest='save_format',
        action="store_const",
        help="Save the checkpoints in .h5 format."
    )
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before starting with the training.'
    )

    parser.add_argument(
        '--verbosity',
        type=int,
        default='1',
        help="How detailed the output should be.")
    parser.add_argument(
        '--seed',
        type=int,
        default='59185',
        help="Seed for the preprocessing operations.")
    
    parser.add_argument(
        '--no_mlflow',
        default=False,
        action="store_true",
        help='Don\'t use MLFlow')

    ARGS, unparsed = parser.parse_known_args()
    main()


