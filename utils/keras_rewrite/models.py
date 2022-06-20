from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, Reshape
from tensorflow.keras import Input, activations, initializers, Sequential

def create_tiny_conv(input_shape, batch_size, label_count:int, filter_settings = (10, 8, 8), dropout_rate:float=0.5, name="tiny_conv"):
  """Builds a convolutional model aimed at microcontrollers.

  Devices like DSPs and microcontrollers can have very small amounts of
  memory and limited processing power. This model is designed to use less
  than 20KB of working RAM, and fit within 32KB of read-only (flash) memory.

  Here's the layout of the graph:

       (input_tensor)
              v
           [Conv2D]<-(weights, bias)
              v
            [ReLu]
              v
          [Dropout]
              v
          [Flatten]
              v
           [Dense]<-(weights, bias)
              v
           [SoftMax]
              v

  This doesn't produce particularly accurate results, but it's designed to be
  used as the first stage of a pipeline, running on a low-energy piece of
  hardware that can always be on, and then wake higher-power chips when a
  possible utterance has been found, so that more accurate analysis can be done.

  During training, a dropout node is introduced after the relu, controlled by a
  placeholder.

  Args:
    input_shape: A keras input tensor that contains audio features.
    batch_size:
    label_count:
    filter_settings:
    dropout_rate:
    name:

  Returns:
    A keras model that has the properties described above.
  """
  return Sequential([
    Input(shape=input_shape, batch_size=batch_size),
    Reshape((49, 40, 1)),
    # Convolutional layer
    Conv2D( filters=filter_settings[-1],
            kernel_size=filter_settings[:2],
            activation=activations.relu,
            strides=(2,2),
            kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.01),\
            bias_initializer=initializers.Zeros(),
            padding='same',
            name='convolutional'
          ),

    # Dropout layer
    Dropout(rate=dropout_rate, name='dropout'),

    # Flatten the output,
    Flatten(),

    # Fully connected layer 
    Dense(  units=label_count,
            activation=activations.softmax,
            kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.01),
            bias_initializer=initializers.Zeros(),
            name='fully_connected'
          )
  ], name = name)

def create_conv(input_shape, batch_size, label_count:int,
  filter_settings_1 = (20, 8, 64), filter_settings_2 = (10, 4, 64),
  dropout_rate_1:float=0.5, dropout_rate_2:float=0.5, name="conv"):
  """
  Here's the layout of the graph:

      (input_tensor)
            v
        [Conv2D]<-(weights, bias)
          v
        [Relu]
          v
      [Dropout]
          v
      [MaxPool]
          v
      [Conv2D]<-(weights, bias)
          v
        [Relu]
          v
      [Dropout]
          v
        [Dense]<-(weights, bias)
          v
      [SoftMax]
          v

  """
  return Sequential([
    Input(shape=input_shape, batch_size=batch_size),
  
    
    # 1st Convolutional layer
    Conv2D( filters=filter_settings_1[-1],
            kernel_size=filter_settings_1[:2],
            activation=activations.relu,
            strides=(1,1),
            kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.01),\
            bias_initializer=initializers.Zeros(),
            padding='same',
            name='convolutional1'
          ),

    # 1st Dropout layer
    Dropout(rate=dropout_rate_1, name='dropout1'),

    MaxPool2D(pool_size=(2, 2),
              strides=(2, 2),
              padding='same'),

    # 2nd Convolutional layer
    Conv2D( filters=filter_settings_2[-1],
            kernel_size=filter_settings_2[:2],
            activation=activations.relu,
            strides=(1,1),
            kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.01),\
            bias_initializer=initializers.Zeros(),
            padding='same',
            name='convolutional2'
          ),

    # 2nd Dropout layer
    Dropout(rate=dropout_rate_2, name='dropout2'),

    # Flatten the output,
    Flatten(),

    # Fully connected layer 
    Dense(  units=label_count,
            activation=activations.softmax,
            kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.01),
            bias_initializer=initializers.Zeros(),
            name='fully_connected'
          )
  ], name = name)

def create_tiny_embedding_conv(input_shape, batch_size, label_count:int,
  filter_settings_1 = (10, 8, 8), filter_settings_2 = (10, 8, 8),
  dropout_rate_1:float=0.5, dropout_rate_2:float=0.5, name="tiny_embedding_conv"):
  
  return Sequential([
    Input(shape=input_shape, batch_size=batch_size),
    
    # 1st Convolutional layer
    Conv2D( filters=filter_settings_1[-1],
            kernel_size=filter_settings_1[:2],
            activation=activations.relu,
            strides=(2,2),
            kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.01),\
            bias_initializer=initializers.Zeros(),
            padding='same',
            name='convolutional1'
          ),

    # 1st Dropout layer
    Dropout(rate=dropout_rate_1, name='dropout1'),

    # 2nd Convolutional layer
    Conv2D( filters=filter_settings_2[-1],
            kernel_size=filter_settings_2[:2],
            activation=activations.relu,
            strides=(8,8),
            kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.01),\
            bias_initializer=initializers.Zeros(),
            padding='same',
            name='convolutional2'
          ),

    # 2nd Dropout layer
    Dropout(rate=dropout_rate_2, name='dropout2'),

    # Flatten the output,
    Flatten(),

    # Fully connected layer 
    Dense(  units=label_count,
            activation=activations.softmax,
            kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.01),
            bias_initializer=initializers.Zeros(),
            name='fully_connected'
          )
  ], name = name)

def create_single_fc(input_shape, batch_size, label_count:int, name="single_fc"):
  return Sequential([
    Input(shape=input_shape, batch_size=batch_size),

    # Flatten the output,
    Flatten(),

    # Fully connected layer 
    Dense(  units=label_count,
            activation=activations.softmax,
            kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.001),
            bias_initializer=initializers.Zeros(),
            name='fully_connected'
          )
  ], name = name)

def create_low_latency_conv(input_shape, batch_size, label_count:int, filter_count=186, dropout_rate=0.5, name="low_latency_conv"):
  return Sequential([
    Input(shape=input_shape, batch_size=batch_size),
    
    # 1st Convolutional layer
    Conv2D( filters=filter_count,
            kernel_size=(input_shape[1], 8),
            activation=activations.relu,
            strides=(1,1),
            kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.01),\
            bias_initializer=initializers.Zeros(),
            padding='valid',
            name='convolutional'
          ),

    # 1st Dropout layer
    Dropout(rate=dropout_rate, name='dropout1'),

    # Flatten the output,
    Flatten(),

    # Fully connected layer 
    Dense(  units=128,
            kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.01),
            bias_initializer=initializers.Zeros(),
            name='fully_connected1'
          ),
        # Fully connected layer 
    Dense(  units=128,
            kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.01),
            bias_initializer=initializers.Zeros(),
            name='fully_connected2'
          ),
        # Fully connected layer 
    Dense(  units=label_count,
            activation=activations.softmax,
            kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.01),
            bias_initializer=initializers.Zeros(),
            name='fully_connected3'
          )
  ], name = name)

def create_low_latency_svdf(input_shape, batch_size, label_count:int, name='low_latency_svdf'):
  pass

def create_model(architecture, input_shape, batch_size, label_count):
  architecture = architecture.lower()
  if architecture == 'conv':
    return create_conv(input_shape, batch_size, label_count)
  elif architecture == 'tiny_conv':
    return create_tiny_conv(input_shape, batch_size, label_count)
  elif architecture == 'tiny_embedding_conv':
    return create_tiny_embedding_conv(input_shape, batch_size, label_count)
  elif architecture == 'single_fc':
    return create_single_fc(input_shape, batch_size, label_count)
  elif architecture == 'low_latency_conv':
    return create_low_latency_conv(input_shape, batch_size, label_count)
  elif architecture == 'low_latency_svdf':
    print("Low Latency SVDF not supported yet, defaulting to low_latency_conv")
    return create_low_latency_conv(input_shape, batch_size, label_count)
    #return create_low_latency_svdf(input_shape, batch_size, label_count)
  else:
    raise Exception("model_architecture argument '" + architecture +
                    "' not recognized, should be one of 'conv', 'tiny_conv', 'tiny_embedding_conv',"
                    "'single_fc', 'low_latency_conv', or 'low_latency_svdf'."
                    )
