
    
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

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)

    # input data and model directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))    
    parser.add_argument('--labels', type=str, default=os.environ.get('SM_CHANNEL_LABELS'))


    args, _ = parser.parse_known_args()
    
    spectrogram_ds = tf.data.experimental.load(args.train)
    train_ds = spectrogram_ds
    val_ds = tf.data.experimental.load(args.val)
    test_ds = tf.data.experimental.load(args.test)
    commands = np.load(args.labels + "/commands.npy")
    
    batch_size = args.batch_size
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    
    
    #Add Dataset.cache and Dataset.prefetch operations to reduce read latency while training the model:
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    
    
    #For the model, you'll use a simple convolutional neural network (CNN), since you have transformed the audio files into spectrogram images.

    # Your tf.keras.Sequential model will use the following Keras preprocessing layers:
    # - tf.keras.layers.Resizing: to downsample the input to enable the model to train faster.
    # - tf.keras.layers.Normalization: to normalize each pixel in the image based on its mean and standard deviation.
    # - For the Normalization layer, its adapt method would first need to be called on the training data in order to compute aggregate statistics (that is, the mean and the standard deviation).
    
    
    for spectrogram, _ in spectrogram_ds.take(1):
      input_shape = spectrogram.shape
    print('Input shape:', input_shape)
    num_labels = len(commands)
    print(num_labels)
    
    for spectrogram, _ in val_ds.take(1):
      tmp = spectrogram.shape
    print('val shape:', tmp)

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    print(model.summary())
    
    #     Configure the Keras model with the Adam optimizer and the cross-entropy loss:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    
    EPOCHS = args.epochs
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    
    # Save model to the SM_MODEL_DIR path
    print("Saving model to {}".format(os.environ.get('SM_MODEL_DIR')))
    model.save(os.environ.get('SM_MODEL_DIR'))
    

