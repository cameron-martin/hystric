import datetime
from pathlib import Path
from typing import Tuple

import tensorflow as tf
import numpy as np
from tensorflow import keras
import kapre
import tensorflow_datasets as tfds
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from hystric.load import load

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# mixed_precision.set_policy(mixed_precision.Policy('mixed_float16'))

SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 32
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

UNITS=64
LSTM_LAYERS=5

def pad_to_1s(audio):
    return tf.pad(audio, [[0, SAMPLE_RATE - tf.shape(audio)[0]]])

def preprocess_example(speech, label):
    return preprocess_audio(speech), preprocess_label(label)

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

def preprocess_label(label: tf.Tensor):
    chars = tf.strings.bytes_split(label)
    return table.lookup(chars)

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

def ds_conv_block(input, dilation):
    x = tf.keras.layers.DepthwiseConv2D(3, padding='same', dilation_rate=(dilation, dilation))(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(NUM_FEATURE_MAPS, 1, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def residual_block(input, dilations):
    x = ds_conv_block(input, dilation=dilations[0])
    x = ds_conv_block(x, dilation=dilations[1])

    return keras.layers.Add()([input, x])

# Merges the training example with the piece of silence with 80% probability
def merge_with_silence(silence, train):
    audio, label = train
    silence_weight = tf.random.uniform(shape=(), minval=0, maxval=0.4)
    merged = tf.add(silence * silence_weight, audio * (1-silence_weight))
    return (merged, label)

class CTCLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=tf.math.count_nonzero(y_true, -1), logit_length=tf.repeat(tf.shape(y_pred)[-1], tf.shape(y_pred)[0]), logits_time_major=False)

def train():
    validation_data, training_data = load(splits=['dev-clean', 'train-clean-100'])

    validation_data = validation_data.map(preprocess_example)
    training_data = training_data.map(preprocess_example)

    training_data = training_data.padded_batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    validation_data = validation_data.padded_batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    # for audio, label in training_data:
    #     print(audio, label)
    #     break

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
    x = tf.keras.layers.Dense(ALPHABET_SIZE, activation='softmax')(x)
    model = keras.Model(input, x)

    model.compile(
        optimizer="adam", loss=CTCLoss(), metrics=[],
    )

    model.summary()

    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest is not None:
        model.load_weights(latest)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINT_FILEPATH), save_weights_only=True
    )

    log_dir = "tmp/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='10, 15')

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    )

    model.fit(
        training_data,
        validation_data=validation_data,
        epochs=100,
        callbacks=[model_checkpoint_callback, tensorboard_callback, early_stopping_callback],
    )
    
    model.save("tmp/model.h5")
