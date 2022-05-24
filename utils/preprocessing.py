
import argparse
import os
import warnings

import pandas as pd
import pathlib
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display


# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
AUTOTUNE = tf.data.AUTOTUNE

input_data_paths = "/opt/ml/processing/input/data"
commands = np.array(tf.io.gfile.listdir(str(input_data_paths)))
commands = commands[commands != 'README.md']
print('Commands:', commands)

def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)

def get_label(file_path):
  parts = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-2]

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=325, fft_length =78 )
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.argmax(label == commands)
  return spectrogram, label_id

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(
      map_func=get_waveform_and_label,
      num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      map_func=get_spectrogram_and_label_id,
      num_parallel_calls=AUTOTUNE)
  return output_ds

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args, _ = parser.parse_known_args()
    
    print("Received arguments {}".format(args))
    
    
    filenames = tf.io.gfile.glob(str(input_data_paths) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
#     print('Number of examples per label:',
#           len(tf.io.gfile.listdir(str(input_data_paths/commands[1]))))
    print('Example file tensor:', filenames[0])
    
    # splitting the data set depending on the training ratio
    
    train_ratio = args.train_ratio
    val_test_ratio = (1 - train_ratio)/2
    
    train_files = filenames[:int(train_ratio*len(filenames))]
    val_files = filenames[int(train_ratio*len(filenames)): int(train_ratio*len(filenames)) + int(val_test_ratio*len(filenames))]
    test_files = filenames[-int(val_test_ratio*len(filenames)):]

    print('Training set size', len(train_files))
    print('Validation set size', len(val_files))
    print('Test set size', len(test_files))
    
    # Converting audio to spectrograms
    train_ds = preprocess_dataset(train_files)
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)
    

    train_features_output_path = "/opt/ml/processing/train"
    val_features_output_path = "/opt/ml/processing/val"
    test_features_output_path = "/opt/ml/processing/test"
    labels_output_path = "/opt/ml/processing/commands/commands"

    
    print("Saving train spectrogram to {}".format(train_features_output_path))
    tf.data.experimental.save(train_ds, train_features_output_path)
   
    print("Saving val spectrogram to {}".format(val_features_output_path))
    tf.data.experimental.save(val_ds, val_features_output_path)
    
    print("Saving test spectrogram to {}".format(test_features_output_path))
    tf.data.experimental.save(test_ds, test_features_output_path)
    
    print("Saving labels to {}".format(labels_output_path))
    np.save(labels_output_path, commands)
    
    
    d = np.load(labels_output_path + ".npy")
    print(commands == d)

    
    
    
    
    


