import argparse
import binascii
import os
import numpy as np
import tensorflow as tf
import utils
from preprocessing import construct_example_dataset
from generate_c_source import generate_c_source

def main():

    if ARGS.h5:
        model = tf.keras.models.load_model(ARGS.input)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(ARGS.input)
    
    if(ARGS.int_only):
        
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        bg_vol_interval = utils.parse_list(ARGS.bg_volume_interval, float)
        if ARGS.data_dir:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            def representative_data_gen():
                for input_value in construct_example_dataset(ARGS.data_dir, element_count=100, sample_rate=ARGS.sample_rate, clip_duration_ms=ARGS.clip_duration_ms,
                window_size_ms=ARGS.window_size_ms, window_stride_ms=ARGS.window_stride_ms, feature_bin_count=ARGS.feature_bin_count, preprocessing_mode=ARGS.preprocess,
                bg_noise_dir=os.path.join(ARGS.data_dir, ARGS.bg_noise_subdir), bg_noise_freq=ARGS.bg_noise_freq, min_bg_noise=bg_vol_interval[0], max_bg_noise=bg_vol_interval[1], time_shift_ms=ARGS.time_shift_ms).batch(1).take(100):
                    flattened_data = np.array(tf.reshape(input_value, [-1]), dtype=np.float32).reshape(1, 1960)
#                     numpy_data = np.array(input_value, dtype=np.float32)
                    yield [flattened_data]
            converter.representative_dataset = representative_data_gen
        
        tflite_model = converter.convert()
        tflite_model_size = open(ARGS.output, "wb").write(tflite_model)
        print("Exported quantized model: %d bytes" % tflite_model_size)
    else:
        float_tflite_model = converter.convert()
        float_tflite_model_size = open(ARGS.output, "wb").write(float_tflite_model)
        print("Exported float model: %d bytes" % float_tflite_model_size)

    if ARGS.cfile:
        generate_c_source(ARGS.output, str(ARGS.output).split('.')[0] + ".c", ARGS.varname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        help='Path to the saved model or h5 file.')
    parser.add_argument(
        '--output',
        type=str,
        help='Where to save the generated tflite model.')
    parser.add_argument(
        '--h5',
        default=False,
        action="store_true",
        help="Read input file in .h5 format."
    )
    parser.add_argument(
        "--cfile",
        default=False,
        action="store_true",
        help="Also generate the corresponding C source file."
    )
    parser.add_argument(
        "--varname",
        type=str,
        default="tflite",
        help="Name of the variable stored in the C file."
    )
    parser.add_argument(
    '--int_only',
    dest='int_only',
    default=False,
    help='Use integer only quantization.',
    action='store_true')
    
    # Preprocessing settings (only needed if --int_only)
    parser.add_argument(
        '--data_dir',
        type=str,
        help="""\
        Where the audio data is located
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
        default=20.0,
        help='How far to move in time between spectrogram timeslices.',
    )
    parser.add_argument(
        '--feature_bin_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint',
    )
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
        '--bg_noise_subdir',
        type=str,
        default='_background_noise_',
        help="""\
            Name or relative path of the subdirectory containing background noise files. The subdirectory
            should be located in 'data_dir'. If a relative path is given, it should be relative to the path of 'data_dir'
        """)



    ARGS, unparsed = parser.parse_known_args()
    main()