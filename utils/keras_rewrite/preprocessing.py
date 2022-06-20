import math
import os.path
import hashlib
import re
import utils
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from tensorflow.python.ops import gen_audio_ops as audio_ops

# If it's available, load the specialized feature generator. If this doesn't
# work, try building with bazel instead of running the Python script directly.
try:
    from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op as frontend_op  # pylint:disable=g-import-not-at-top
except ImportError:
    frontend_op = None

RNG = np.random.default_rng(seed=59185)

def decode_wav_file(wav_path, sample_rate=-1, clip_duration_ms=-1, channels=1):
    """Reads and decodes a wav file and returns the resulting audio tensor.

    Args:
        wav_path:           Path to the directory containing the audio data
        sample_rate:        Target sample rate. If not specified load and decode all samples.
        clip_duration_ms:   Clip duration in miliseconds. If not specified load and decode all samples.
        channels:           Desired channel count. If not specified the audio tensor will contain a single channel.

    Returns:
        The decoded audio tensor. The final sample count will be sample_rate * clip_duration_ms / 1000.

    Raises:
        Exception: If the file doesn't exist.
    """
    if not utils.path_exists(wav_path):
        raise Exception("File " + wav_path + " does not exist.")

    samples = -1 if sample_rate < 0 or clip_duration_ms < 0 else int(sample_rate * clip_duration_ms / 1000)
    return audio_ops.decode_wav(tf.io.read_file(wav_path), desired_channels=channels, desired_samples=samples).audio

def decode_wavs(audio_dir, sample_rate=-1, clip_duration_ms=-1):
    """Searches a folder for audio data, and decodes and loads all found wav files into memory.

    Args:
        audio_dir:          Path to the directory containing the audio data
        sample_rate:        Target sample rate. If not specified load and decode all samples.
        clip_duration_ms:   Clip duration in miliseconds. If not specified load and decode all samples.

    Returns:
        List of raw PCM-encoded audio samples.

    Raises:
        Exception: If files aren't found in the folder.
    """
    data = []
    if not tf.io.gfile.exists(audio_dir):
        raise Exception("Directory " + audio_dir + " does not exist.")

    for wav_path in tf.io.gfile.glob(os.path.join(audio_dir, '*.wav')):
        data.append(decode_wav_file(wav_path, sample_rate, clip_duration_ms))

    if not data:
        raise Exception("No wav files were found in " + audio_dir)

    return data

def set_seed(seed):
    """Sets the seed for all random operations. Using the same seed will provide deterministic behaviour.

    Args:
        seed:   A seed to initialize the random number generator.
    """
    global RNG
    RNG = np.random.default_rng(seed)

def add_time_shift(audio_tensor, time_shift_ms:float):
    """Randomly selects a number in the interval [-time_shift_ms, time_shift_ms] and shifts the audio tensor by this amount in miliseconds
    
    Args:
        audio_tensor:   Input Tensor
        time_shift_ms:  A positive floating point number specifying the size of the interval [-time_shift_ms, time_shift_ms].
    
    Returns:
        The audio tensor produced by shifting the input tensor by a random amount.
    
    Raises:
        Exception: If time_shift_ms is less than or equal to zero.

    """
    samples = audio_tensor.shape[0]
    if time_shift_ms <= 0:
        raise Exception("Time shift argument '" + str(time_shift_ms) +"' must be greater than 0.")
    time_shift_amount = RNG.integers(-time_shift_ms, time_shift_ms)

    if time_shift_amount > 0:
        return tf.pad(tensor=audio_tensor,
            paddings=[[time_shift_amount, 0], [0, 0]],
            mode='CONSTANT')[:samples, :]
    else:
        return tf.pad(tensor=audio_tensor,
            paddings=[[0, -time_shift_amount], [0, 0]],
            mode='CONSTANT')[-time_shift_amount:samples-time_shift_amount, :]

def add_bg_noise(audio_tensor, bg_noise_tensors, min_bg_noise:float=0, max_bg_noise:float=1):
    """Randomly selects one of the provided background noises, scales it by a random amount in the specified interval
    and adds it to the provided audio tensor.
    
    Args:
        audio_tensor:       Input Tensor
        bg_noise_tensors:   An array-like object containing the background noise tensors.
        min_bg_noise:       A floating point number in the interval [0, 1], specifying how loud the background noise should be at least. Defaults to 0.
        max_bg_noise:       A floating point number in the interval [0, 1], specifying how loud the background noise should be at most. Defaults to 1.

    Returns:
        The audio tensor produced by shifting the input tensor by a random amount.
    
    Raises:
        Exception: If min_bg_noise or max_bg_noise is not in the interval [0, 1] or if min_bg_noise is greater than max_bg_noise.

    """
    if min_bg_noise < 0 or min_bg_noise > 1 or max_bg_noise < 0 or max_bg_noise > 1 or min_bg_noise > max_bg_noise:
        raise Exception("The specified background noise interval [" + str(min_bg_noise) + ", " + str(max_bg_noise) + "] is not valid")

    samples = audio_tensor.shape[0]
    background_samples = RNG.choice(np.array(bg_noise_tensors, dtype=object))

    if len(background_samples) <= samples:
        raise ValueError(
            'Background sample is too short! Need more than %d'
            ' samples but only %d were found' %
            (samples, len(background_samples)))
    background_offset = np.random.randint(
        0, len(background_samples) - samples)
    background_clipped = background_samples[background_offset:(
        background_offset + samples)]
    
    bg_noise = tf.reshape(background_clipped, [samples, 1])
    bg_noise_volume = RNG.uniform(min_bg_noise, max_bg_noise)

    return tf.clip_by_value(tf.add(tf.multiply(bg_noise, bg_noise_volume), audio_tensor), -1.0, 1.0)

def audio_to_spectrogram(audio_tensor, sample_rate:int,
                         window_size_ms:float, window_stride_ms:float, feature_bin_count:int, preprocessing_mode:str='mfcc'):
    """
    
    Args:

    Returns:

    Raises:

    """
    window_size_samples = int((sample_rate * window_size_ms) / 1000)
    window_stride_samples = int((sample_rate * window_stride_ms) / 1000)

    spectrogram = audio_ops.audio_spectrogram(audio_tensor,
                                              window_size=window_size_samples,
                                              stride=window_stride_samples,
                                              magnitude_squared=True)

    # The number of buckets in each FFT row in the spectrogram will depend on
    # how many input samples there are in each window. This can be quite
    # large, with a 160 sample window producing 127 buckets for example. We
    # don't need this level of detail for classification, so we often want to
    # shrink them down to produce a smaller result. That's what this section
    # implements. One method is to use average pooling to merge adjacent
    # buckets, but a more sophisticated approach is to apply the MFCC
    # algorithm to shrink the representation.
    if preprocessing_mode == 'average':
        fft_bin_count = 1 + (utils.next_power_of_two(window_size_samples) / 2)
        average_window_width = int(math.floor(fft_bin_count / feature_bin_count))
        output = tf.nn.pool(
            input=tf.expand_dims(spectrogram, -1),
            window_shape=[1, average_window_width],
            strides=[1, average_window_width],
            pooling_type='AVG',
            padding='SAME')
            
    elif preprocessing_mode == 'mfcc':
        output = tf.transpose(audio_ops.mfcc(
            spectrogram,
            sample_rate,
            dct_coefficient_count=feature_bin_count), [1,2,0])
            # -> (1, 49, 40) -> (49, 40, 1)
    else:
        if not frontend_op:
            raise Exception("""
                Micro frontend op is currently not available when running TensorFlow 
                directly from Python. You need to build and run through Bazel.
                """)
        int16_input = tf.cast(tf.multiply(audio_tensor, 32768), tf.int16)
        micro_frontend = frontend_op.audio_microfrontend(
            int16_input,
            sample_rate=sample_rate,
            window_size=window_size_ms,
            window_step=window_stride_ms,
            num_channels=feature_bin_count,
            out_scale=1,
            out_type=tf.float32)
        output = tf.expand_dims(tf.multiply(micro_frontend, (10.0 / 256.0)), -1)
        ##### (49, 40, 1)

    return output


def create_data_index(data_dir:str, valid_labels:list, val_pc:int=33, test_pc:int=33, unknown_pc:int = -1, silence_pc:int = 0, ignore:list=[], verbose:bool=True):
    """Creates a dictionary referencing the files contained in 'data_dir', organized by partition and label.

    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the 'data_dir', figures out the
    right labels for each file based on the name of the subdirectory it belongs to,
    and uses a stable hashing algorithm to assign it to a partition.

    The directory should be formatted as:
        data_dir/
            label_1/
                sample_1.wav
                sample_2.wav
                sample_3.wav
                ...
            .
            .
            .
            label_n/
                ...
            ...
            ignore_dir1/
            ignore_dir2/
            ...

    Args:
        data_dir:       Path of the data directory.
        valid_labels:   Labels of the classes we want to be able to recognize. The labels '_silence_' and '_unknown_' will be added automatically.
        val_pc:         How much of the dataset to use for validation. The validation partition will contain this percentage of the original data.
        test_pc:        How much of the dataset to use for testing. The testing partition will contain this percentage of the original data. The remaining percentage will be put into the training partition.
        unknown_pc:     How much of the data should consist of audio labeled as unknown. Defaults to -1 to include all unknown data.
        silence_pc:     How much of the resulting data should be background noise. Defaults to 0.
        ignore:         List of the names or paths of the subdirectories in data_dir containing data that should not be stored in the returned index. 
                        Files in this subdirectory will be ignored. If you plan on using background noise, include the background noise directory in this list.
        
    Returns:
      Dictionary containing a list of (file, label) pairs for each dataset partition,
      A lookup map for each class to determine its numeric index,
      A list containing all differentiable labels.

    Raises:
      Exception: If expected files are not found.
    """

    data_index = {'validation': [], 'testing': [], 'training': []}
    unknown_index = {'validation': [], 'testing': [], 'training': []}
    all_words = []

    ignore = list(map(os.path.basename, ignore))

    # Look through all the subfolders to find audio samples
    search_path = os.path.join(data_dir, '*', '*.wav')
    for wav_path in tqdm(tf.io.gfile.glob(search_path), desc="Gathering .wav files from " + data_dir + " ", disable=not verbose):
        # Get the parent directory name of the file 'wav_path'
        word = os.path.basename(os.path.dirname(wav_path)).lower()
        # Treat the '_background_noise_' folder as a special case, since we expect
        # it to contain long audio samples we mix in to improve training.
        if word in ignore:
            continue
        if word not in all_words:
            all_words.append(word)
        partition = get_partition(wav_path, val_pc, test_pc)
        # If it's a known class, store its detail, otherwise add it to the list
        # we'll use to train the unknown label.
        if word in valid_labels:
            data_index[partition].append((wav_path, word))
        else:
            unknown_index[partition].append((wav_path, word))
        
    if not all_words:
        raise Exception("No .wavs found at " + search_path)
    for label in valid_labels:
        if label not in all_words:
            raise Exception("Expected to find " + label +
                            " in labels but only found " +
                            ", ".join(all_words))

    # We need an arbitrary file to load as the input for the silence samples.
    # It's multiplied by zero later, so the content doesn't matter.
    silence_wav_path = data_index['training'][0][0]
    for partition in ['validation', 'testing', 'training']:
        partition_size = len(data_index[partition])


        # Pick some unknowns to add to each partition of the data set.
        RNG.shuffle(unknown_index[partition])
        unknown_size = None if unknown_pc < 0 else int(math.ceil(partition_size * unknown_pc / (100 - unknown_pc)))
        silence_size = int(math.ceil(partition_size * silence_pc / (100 - silence_pc)))
        
        data_index[partition].extend(unknown_index[partition][:unknown_size])

        data_index[partition].extend([(silence_wav_path, '_silence_')]*silence_size)

    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
        RNG.shuffle(data_index[set_index])

    # Prepare the rest of the result data structure.
    label_to_index = {}
    for word in all_words:
        if word in valid_labels:
            label_to_index[word] = valid_labels.index(word) + 2
        else:
            label_to_index[word] = 1
    label_to_index['_silence_'] = 0

    # Prepare list of all labels that will be used for training.
    all_labels = ['_silence_', '_unknown_'] + valid_labels
    
    return data_index, label_to_index, all_labels

def get_partition(file_path, val_pc:int, test_pc:int, max_files_per_partition = 2**27 - 1, ignore_infix = '_nohash_'):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
        filename:                   File path of the data sample.
        val_pc:                     How much of the data set to use for validation.
        test_pc:                    How much of the data set to use for testing.
        max_files_per_partition:    Maximum number of files per partition, used for calculating the hash values. Defaults to 2**27 - 1.
        ignore_infix:               Anything after ignore_infix in a filename will be ignored for set determination. 
                                    Defaults to '_nohash_'. For example the files 'dummyfile_nohash_0.wav' and
                                    'dummyfile_nohash_1.wav' will be placed in the same partition
    Returns:
        String, one of 'training', 'validation', or 'testing'.
    """
    if val_pc + test_pc >= 100:
        raise Exception("val_pc + test_pc should be smaller than 100")
    if val_pc < 0 or test_pc < 0:
        raise Exception("val_pc and test_pc should be positive") 

    # We want to ignore anything after ignore_infix in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(re.escape(ignore_infix) + r'.*$', '',  os.path.basename(file_path))

    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hashed = hashlib.sha1(hash_name.encode('utf8')).hexdigest()
    percentage_hash = ((int(hashed, 16) %
                        (max_files_per_partition + 1)) *
                        (100.0 / max_files_per_partition))
    if percentage_hash < val_pc:
        return 'validation'
    elif percentage_hash < (test_pc + val_pc):
        return 'testing'
    else:
        return 'training'

def construct_datasets(data_dict:dict, label_to_index:dict, sample_rate:int, clip_duration_ms:int,
                        window_size_ms:float=0, window_stride_ms:float=0, feature_bin_count:int=0, preprocessing_mode:str='mfcc',
                        bg_noise_dir:str='', bg_noise_freq:float=0, min_bg_noise:float=0, max_bg_noise:float=1,
                        time_shift_ms:float=0, verbose:bool=True):
    """
        Preprocesses the files referenced in the data dictionary and generates training, validation and testing datasets.


        Args:
            data_dict:          A dictionary with the keys 'training', 'validation' and 'training'. The corresponding values should be lists of (file_path, label) pairs.
            label_to_index:     A dictionary to convert labels to integer values. Labels that should be indifferentiable should map to the same index.
            sample_rate:        Target sample rate of the audio files.
            clip_duration_ms:   Clip duration in miliseconds.

            (Spectrogram)
            window_size_ms:     
            window_stride_ms:   
            feature_bin_count:  How many bins to use for the MFCC fingerprint.
            preprocessing_mode: 

            (Background Noise)
            bg_noise_dir:
            bg_noise_freq:
            min_bg_noise:
            max_bg_noise:

            (Time Shift)
            time_shift_ms:      



        Returns:
            Dictionary containing the training, validation and testing datasets.
        
    """
    if not not verbose:
        print("Reading data index...")
        for p, d in data_dict.items():
            print(p.capitalize() + " partition: " + str(len(d)) + " entries")

    do_ts = bool(time_shift_ms)
    do_bg = all([bg_noise_dir, bg_noise_freq])
    do_sp = all([window_size_ms, window_stride_ms, feature_bin_count])
    
    # Load background noise directory into memory
    if do_bg:
        bg_noise_data = decode_wavs(bg_noise_dir)

    id_op = lambda x: x
    decode_op = lambda p: (decode_wav_file(p[0], sample_rate, clip_duration_ms), p[1])
    silence_and_label_op = lambda p: (tf.multiply(p[0], 0) if p[1] == '_silence_' else p[0], label_to_index[p[1]])
    time_shift_op = (lambda p: (add_time_shift(p[0], time_shift_ms), p[1])) if do_ts else id_op
    bg_noise_op = (lambda p: (add_bg_noise(p[0], bg_noise_data, min_bg_noise, max_bg_noise), p[1])) if do_bg else id_op
    spectrogram_op = (lambda p: (audio_to_spectrogram(p[0], sample_rate, window_size_ms, window_stride_ms, feature_bin_count, preprocessing_mode), p[1])) if do_sp else id_op

    op = utils.compose([spectrogram_op, bg_noise_op, time_shift_op, silence_and_label_op, decode_op])

    return {partition : tf.data.Dataset.from_tensor_slices(tuple(list(l) for l in zip(*map(op, tqdm(dataset, desc=("Generating " + partition + " partition"), disable=not verbose))))) for (partition, dataset) in data_dict.items()}

def construct_example_dataset(data_dir, element_count, sample_rate:int, clip_duration_ms:int,
                            window_size_ms:float=0, window_stride_ms:float=0, feature_bin_count:int=0, preprocessing_mode:str='mfcc',
                            bg_noise_dir:str='', bg_noise_freq:float=0, min_bg_noise:float=0, max_bg_noise:float=1,
                            time_shift_ms:float=0, ignore=[]):
    data_index = []
    search_path = os.path.join(data_dir, '*', '*.wav')
    ignore.append(bg_noise_dir)
    ignore = list(map(os.path.basename, ignore))
    
    for wav_path, _ in tqdm(zip(tf.io.gfile.glob(search_path), range(element_count)), desc="Gathering .wav files from " + data_dir + " "):
        # Get the parent directory name of the file 'wav_path'
        word = os.path.basename(os.path.dirname(wav_path)).lower()
        # Treat the '_background_noise_' folder as a special case, since we expect
        # it to contain long audio samples we mix in to improve training.
        if word in ignore:
            continue
        data_index.append(wav_path)

    do_ts = bool(time_shift_ms)
    do_bg = all([bg_noise_dir, bg_noise_freq])
    do_sp = all([window_size_ms, window_stride_ms, feature_bin_count])

    # Load background noise directory into memory
    if do_bg:
        bg_noise_data = decode_wavs(bg_noise_dir)

    id_op = lambda x: x
    decode_op = lambda p: decode_wav_file(p, sample_rate, clip_duration_ms)
    time_shift_op = (lambda p: add_time_shift(p, time_shift_ms)) if do_ts else id_op
    bg_noise_op = (lambda p: add_bg_noise(p, bg_noise_data, min_bg_noise, max_bg_noise)) if do_bg else id_op
    spectrogram_op = (lambda p: audio_to_spectrogram(p, sample_rate, window_size_ms, window_stride_ms, feature_bin_count, preprocessing_mode)) if do_sp else id_op

    op = utils.compose([spectrogram_op, bg_noise_op, time_shift_op, decode_op])

    return tf.data.Dataset.from_tensor_slices(list(map(op, data_index)))


@tf.function
def wav_to_spectrogram(wav_data, sample_rate, clip_duration_ms, window_size_ms, window_stride_ms, feature_bin_count, preprocessing_mode='mfcc'):
    sample_rate_ms = sample_rate / 1000
    samples = -1 if sample_rate < 0 or clip_duration_ms < 0 else sample_rate_ms * clip_duration_ms
    window_size_samples = sample_rate_ms * window_size_ms
    window_stride_samples = sample_rate_ms * window_stride_ms

    decoded_sample_data = tf.audio.decode_wav(
        wav_data,
        desired_channels=1,
        desired_samples=samples)
    spectrogram = audio_ops.audio_spectrogram(decoded_sample_data.audio,
                                              window_size=window_size_samples,
                                              stride=window_stride_samples,
                                              magnitude_squared=True)
    if preprocessing_mode == 'average':
        fft_bin_count = 1 + (utils.next_power_of_two(window_size_samples) / 2)
        average_window_width = tf.floor(fft_bin_count / feature_bin_count)
        output = tf.nn.pool(
            input=tf.expand_dims(spectrogram, -1),
            window_shape=[1,average_window_width],
            strides=[1, average_window_width],
            pooling_type='AVG',
            padding='SAME')
    elif preprocessing_mode == 'mfcc':
        output = tf.transpose(audio_ops.mfcc(
            spectrogram,
            sample_rate,
            dct_coefficient_count=feature_bin_count), [1,2,0])
    elif preprocessing_mode == 'micro':
        window_size_ms = (window_size_samples *
                        1000) / sample_rate
        window_step_ms = (window_stride_samples *
                        1000) / sample_rate
        int16_input = tf.cast(
            tf.multiply(decoded_sample_data.audio, 32767), tf.int16)
        micro_frontend = frontend_op.audio_microfrontend(
            int16_input,
            sample_rate=sample_rate,
            window_size=window_size_ms,
            window_step=window_step_ms,
            num_channels=feature_bin_count,
            out_scale=1,
            out_type=tf.float32)
        output = tf.expand_dims(tf.multiply(micro_frontend, (10.0 / 256.0)), -1)
        
    else:
        raise Exception('Unknown preprocess mode "%s" (should be "mfcc",'
                        ' "average", or "micro")' % (preprocessing_mode))
    output = tf.reshape(output, [-1])
    return output