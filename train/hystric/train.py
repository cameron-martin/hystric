import datetime
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
import kapre
import tensorflow_datasets as tfds

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


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

def with_residual_connection(input, layer):
    return keras.layers.Add()([input, layer(input)])

def preprocess_data(values):
    return (tf.cast(tf.divide(values['audio'], PCM_16_MAX), 'float32'), tf.one_hot(values['label'], NUM_CATEGORIES))

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


def train():
    result: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset] = tfds.load('speech_commands', shuffle_files=True, split=['train', 'validation', 'test'])

    train_data, validation_data, test_data = result

    train_data = train_data.map(preprocess_data).shuffle(SHUFFLE_BUFFER_SIZE).map(apply_random_transformation).padded_batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    validation_data = validation_data.map(preprocess_data).padded_batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    # iter = train_data.__iter__()
    # print(next(iter))
    # print(next(iter))
    # return

    input = keras.Input(shape=(None,))

    x = keras.layers.Lambda(lambda input: tf.expand_dims(input, -1), input_shape=(None,))(input) # Add channels layer
    x = kapre.composed.get_melspectrogram_layer(return_decibel=True, input_shape=(None,1), sample_rate=SAMPLE_RATE, win_length=ms_to_samples(30), hop_length=ms_to_samples(10), input_data_format='channels_last', output_data_format='channels_last')(x)
    x = kapre.signal.LogmelToMFCC(n_mfccs=40, data_format='channels_last')(x)
    # x = tf.keras.layers.AveragePooling2D(pool_size=(4, 3), data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(NUM_FEATURE_MAPS, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = residual_block(x, dilations=(1, 1))
    x = residual_block(x, dilations=(1, 2))
    x = residual_block(x, dilations=(2, 2))
    x = residual_block(x, dilations=(4, 4))
    x = residual_block(x, dilations=(4, 8))
    x = residual_block(x, dilations=(8, 8))
    x = residual_block(x, dilations=(16, 16))
    x = ds_conv_block(x, dilation=16)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')(x)
    model = keras.Model(input, x)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=[keras.metrics.CategoricalAccuracy()],
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
        train_data,
        validation_data=validation_data,
        epochs=100,
        callbacks=[model_checkpoint_callback, tensorboard_callback, early_stopping_callback],
    )

    model.save("tmp/model.h5")
