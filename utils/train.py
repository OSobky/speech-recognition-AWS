
    
import argparse
import os
import warnings
import itertools
import io

import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display



class EvaluationCallback(keras.callbacks.Callback):
    def on_test_end(self, logs=None):
        self.log_confusion_matrix()

    def plot_confusion_matrix(self, cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
        
        Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
        """
        
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Normalize the confusion matrix.
        cm = cm.numpy()
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        
        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
            
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # plt.show()
        return figure

    def plot_to_image(self, figure):
        """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        """
        
        

        buf = io.BytesIO()
        # Use plt.savefig to save the plot to a PNG in memory.
        plt.savefig(buf, format='png')
        
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        

        # Use tf.image.decode_png to convert the PNG buffer
        # to a TF image. Make sure you use 4 channels.
        image = tf.image.decode_image(buf.getvalue(), channels=4)
        # print(base64.b64decode((buf.getvalue())))
        
        # Use tf.expand_dims to add the batch dimension
        image = tf.expand_dims(image, 0)
        
        return image

    def log_confusion_matrix(self):
        
        # Use the model to predict the values from the test_images.
        
        test_audio = []
        test_labels = []

        for audio, label in test_ds:
            print(audio.shape)
            break
            
        for audio, label in test_ds:
            test_audio.append(audio.numpy())
            test_labels.append(label.numpy())
            
        test_audio = np.asarray(test_audio).astype('float32')
        test_labels = np.asarray(test_labels).astype('float32')
        
        print("test_audio shape : ", test_audio.shape)
        
        y_pred = np.argmax(model.predict(test_audio), axis=1)
        y_true = test_labels

        # Calculate the confusion matrix using sklearn.metrics
        cm = tf.math.confusion_matrix(y_true, y_pred)
        print(cm)
        
        figure = self.plot_confusion_matrix(cm, class_names=['yes', 'no'])
        cm_image = self.plot_to_image(figure)
        # print(cm_image)
        
        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=0)

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
    test_ds_batch = test_ds.batch(batch_size)
    
    
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
        layers.Conv2D(8, 3, activation='relu'),
        #layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        #layers.Dense(128, activation='relu'),
        #layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    print(model.summary())
    
    #     Configure the Keras model with the Adam optimizer and the cross-entropy loss:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    
    LOG_DIR = "/opt/ml/output/tensorboard"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
    file_writer_cm = tf.summary.create_file_writer(LOG_DIR + '/cm')
    
    
    EPOCHS = args.epochs
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[tf.keras.callbacks.EarlyStopping(verbose=1, patience=2), tensorboard_callback],
    )
    
    # Creating Confusion matrix using test_ds   
    
    # Define the per-epoch callback.
    
    result = model.evaluate(
                            test_ds_batch,
                            callbacks=[tensorboard_callback, EvaluationCallback()]
                           )
    print("test set results:", dict(zip(model.metrics_names, result)))
    
    # Save model to the SM_MODEL_DIR path
    model_path = os.environ.get('SM_MODEL_DIR') + "/1"
    print("Saving model to {}".format(model_path))
    model.save(model_path)
    

