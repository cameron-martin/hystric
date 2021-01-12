import datetime
from pathlib import Path
from typing import Tuple

import tensorflow as tf
import numpy as np
from tensorflow import keras
import kapre
import tensorflow_datasets as tfds
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from hystric.load.librispeech import load_librispeech
from hystric.load.cmu_dictionary import load_cmu

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# mixed_precision.set_policy(mixed_precision.Policy('mixed_float16'))

SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 8
NUM_CATEGORIES = 12
NUM_FEATURE_MAPS = 64
CHECKPOINT_FILEPATH = Path("tmp/checkpoint/cp-{epoch:04d}.ckpt")
CHECKPOINT_DIR = CHECKPOINT_FILEPATH.parent
PCM_16_MAX = 2**15
SAMPLE_RATE = 16000

def ms_to_samples(ms):
    return int((SAMPLE_RATE/1000) * ms)

MFCC_WIDTH=ms_to_samples(25)
MFCC_HOP=ms_to_samples(10)
FRAME_WIDTH=5
FRAME_HOP=3

SAMPLES_PER_FRAME = (FRAME_WIDTH - 1) * MFCC_HOP + MFCC_WIDTH
SAMPLES_PER_HOP = MFCC_HOP * FRAME_HOP

UNITS=128
LSTM_LAYERS=5

def pad_to_1s(audio):
    return tf.pad(audio, [[0, SAMPLE_RATE - tf.shape(audio)[0]]])

def preprocess_example(speech, label, pronouncing_dictionary_index, pronouncing_dictionary_values):
    return preprocess_audio(speech), preprocess_label(label, pronouncing_dictionary_index, pronouncing_dictionary_values)

def preprocess_audio(audio):
    '''Convert PCM to normalised floats and chunk audio into frames of correct size to feed to RNN'''
    return tf.signal.frame(tf.cast(audio, 'float32') / PCM_16_MAX, frame_length=SAMPLES_PER_FRAME, frame_step=SAMPLES_PER_HOP)

alphabet = 'abcdefghijklmnopqrstuvwxyz\' '
ALPHABET_SIZE = len(alphabet) + 1
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(tf.strings.bytes_split(alphabet)),
        values=tf.constant(range(1, ALPHABET_SIZE))),
    default_value=-1)

def preprocess_label(label: tf.Tensor, pronouncing_dictionary_index: tf.lookup.StaticHashTable, pronouncing_dictionary_values: tf.RaggedTensor):
    word_indices = pronouncing_dictionary_index.lookup(tf.strings.split(tf.strings.upper(label)))
    return tf.gather(pronouncing_dictionary_values, word_indices).merge_dims(0, 1)

# TODO: use tf.sequence_mask here.
def get_mask(length, shift_amount):
    return tf.cast(tf.logical_and(tf.range(0, length) >= shift_amount, tf.range(0, length) < length + shift_amount), 'float32')

def apply_random_shift(audio):
    max_shift_magnitude = ms_to_samples(100)
    shift_amount = tf.random.uniform(shape=(), minval=-max_shift_magnitude, maxval=max_shift_magnitude, dtype=tf.dtypes.int32)
    audio = tf.roll(audio, shift=shift_amount, axis=0)
    audio = tf.multiply(audio, get_mask(tf.shape(audio)[0], shift_amount))
    return audio

def apply_random_transformation(audio, label):
    audio = apply_random_shift(audio)
    return (audio, label)

# Merges the training example with the piece of silence with 80% probability
def merge_with_silence(silence, train):
    audio, label = train
    silence_weight = tf.random.uniform(shape=(), minval=0, maxval=0.4)
    merged = tf.add(silence * silence_weight, audio * (1-silence_weight))
    return (merged, label)

class CTCLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        tf.print(tf.math.count_nonzero(y_true, -1), summarize=-1, output_stream='file:///project/tmp/print.txt')
        return tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=tf.math.count_nonzero(y_true, -1), logit_length=tf.repeat(tf.shape(y_pred)[-1], tf.shape(y_pred)[0]), logits_time_major=False)

def CTCEditDistance(beam_width=100):
    def ctc_edit_distance(y_true, y_pred):
        sequence_length = tf.repeat(tf.shape(y_pred)[1], tf.shape(y_pred)[0])
        # Transform blank_index from 0 to num_classes - 1 to make up for failure in API. See https://github.com/tensorflow/tensorflow/issues/42993
        y_pred_shifted = tf.roll(y_pred, shift=-1, axis=2)
        decoded, log_probability = tf.nn.ctc_beam_search_decoder(tf.transpose(y_pred_shifted, (1, 0, 2)), sequence_length=sequence_length, beam_width=beam_width, top_paths=1)
        decoded = decoded[0]
        # This undoes the above shift
        num_classes = tf.shape(y_pred)[2]
        decoded = tf.sparse.map_values(lambda value: tf.math.floormod(value + 1, tf.cast(num_classes, 'int64')), decoded)
        # TODO: Work out why this gets cast to a float
        y_true_sparse = tf.sparse.from_dense(tf.cast(y_true, 'int64'))
        is_nonzero = tf.not_equal(y_true_sparse.values, 0)
        y_true_sparse = tf.sparse.retain(y_true_sparse, is_nonzero)
        tf.print(
            tf.sparse.to_dense(y_true_sparse),
            # y_pred,
            tf.sparse.to_dense(decoded),
            summarize=-1,
            output_stream='file:///project/tmp/print.txt',
            sep='\n\n',
            end='\n\n\n\n')
        return tf.edit_distance(decoded, y_true_sparse)
    return ctc_edit_distance



def train():
    validation_data, training_data_100, training_data_360 = load_librispeech(splits=['dev-clean', 'train-clean-100', 'train-clean-360'])
    training_data = training_data_100.concatenate(training_data_360)
    # training_data = training_data.filter(lambda audio, label: tf.shape(audio)[0] < SAMPLE_RATE * 3)

    pronouncing_dictionary, phoneme_mapping = load_cmu()

    keys = tf.constant(list(pronouncing_dictionary.keys()))
    pronouncing_dictionary_index = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=keys,
            values=tf.range(1, tf.shape(keys) + 1)),
        default_value=0)

    pronouncing_dictionary_values = tf.ragged.constant([[]] + list(pronouncing_dictionary.values()))

    def _preprocess_example(speech, label):
        return preprocess_example(speech, label, pronouncing_dictionary_index, pronouncing_dictionary_values)

    validation_data = validation_data.map(_preprocess_example)
    training_data = training_data.map(_preprocess_example)

    # print(phoneme_mapping)

    # for audio, processed_label, label in training_data:
    #     print(label, processed_label)
    #     break

    # return

    training_data = training_data.shuffle(SHUFFLE_BUFFER_SIZE).padded_batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    validation_data = validation_data.padded_batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)


    # result: Tuple[tf.data.Dataset, tf.data.Dataset] = tfds.load('librispeech', shuffle_files=True, split=['train_clean360', 'dev_clean'], as_supervised=True)

    # train_data, validation_data = result


    # train_silence = train_data\
    #     .filter(lambda example: example['label'] == 10)\
    #     .map(lambda example: preprocess_audio(example['audio']))\
    #     .cache()
    # train_data = train_data.map(preprocess_example).shuffle(SHUFFLE_BUFFER_SIZE).map(apply_random_transformation)
    # train_data = tf.data.Dataset.zip((train_silence.repeat().shuffle(SHUFFLE_BUFFER_SIZE), train_data)).map(merge_with_silence)
    # train_data = train_data.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    # validation_data = validation_data.map(preprocess_example).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    # lengths = {}
    # for silence in train_data:
    #     length = tf.shape(silence[0])[0].numpy()
    #     if length not in lengths:
    #         lengths[length] = 0
    #     lengths[length] += 1
    
    #     # play_obj = sa.play_buffer(tf.expand_dims(data['audio'], -1).numpy().astype(np.int16), 1, 2, 16000)
    #     # play_obj.wait_done()

    # print(lengths)


    input = keras.Input(shape=(None, SAMPLES_PER_FRAME))

    n_fft=2048

    x = tf.keras.layers.Lambda(lambda input: tf.signal.stft(
        input,
        fft_length=n_fft,
        frame_length=MFCC_WIDTH,
        frame_step=MFCC_HOP), name="stft")(input)
    x = kapre.Magnitude()(x)
    x = tf.keras.layers.Lambda(lambda input: tf.matmul(input, tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40,
        num_spectrogram_bins=n_fft // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=200,
        upper_edge_hertz=4000)), name="linear_to_mel")(x)
    x = tf.keras.layers.Lambda(lambda input: tf.math.log(input + 1e-6), name="magitude_to_db")(x)
    x = tf.keras.layers.Lambda(lambda input: tf.signal.mfccs_from_log_mel_spectrograms(input), name="mfcc")(x)
    # kapre.composed.get_melspectrogram_layer()
    # x = tf.keras.layers.AveragePooling2D(pool_size=(4, 3), data_format='channels_last')(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    x = tf.keras.layers.Dense(UNITS, activation='tanh')(x)
    for _ in range(LSTM_LAYERS):
        x = tf.keras.layers.LSTM(UNITS, return_sequences=True)(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(ALPHABET_SIZE, activation='softmax'))(x)
    model = keras.Model(input, x)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(), loss=CTCLoss(), metrics=[CTCEditDistance(beam_width=100)],
    )

    model.summary()

    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest is not None:
        model.load_weights(latest)

    callbacks = []

    callbacks.append(keras.callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINT_FILEPATH), save_weights_only=True
    ))

    if True:
        log_dir = "tmp/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0))

    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    ))

    model.fit(
        training_data,
        validation_data=validation_data,
        epochs=100,
        callbacks=callbacks,
    )
    
    model.save("tmp/model.h5")
