import datetime
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from hystric.load.librispeech import load_librispeech
from hystric.load.cmu_dictionary import load_cmu
from hystric.model import compile_model, create_model, SAMPLES_PER_FRAME, SAMPLES_PER_HOP
from hystric.preprocessing import pcm16_to_float32

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# mixed_precision.set_policy(mixed_precision.Policy('mixed_float16'))

SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 64
CHECKPOINT_FILEPATH = Path("tmp/checkpoint/cp-{epoch:04d}.ckpt")
CHECKPOINT_DIR = CHECKPOINT_FILEPATH.parent

def filter_empty(speech, label):
    '''Filter empty labels. This is necessary to avoid an infinite relative edit distance'''
    return tf.not_equal(tf.size(label), 0)

def preprocess_example(speech, label, char_table):
    return preprocess_audio(speech), preprocess_label(label, char_table)

def preprocess_audio(audio):
    '''Convert PCM to normalised floats and chunk audio into frames of correct size to feed to RNN'''
    return tf.signal.frame(pcm16_to_float32(audio), frame_length=SAMPLES_PER_FRAME, frame_step=SAMPLES_PER_HOP)

def preprocess_label(label: tf.Tensor, char_table: tf.lookup.StaticHashTable):
    return char_table.lookup(tf.strings.bytes_split(tf.strings.upper(label)))

ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ\' ')

def train():
    validation_data, training_data_100, training_data_360 = load_librispeech(splits=['dev-clean', 'train-clean-100', 'train-clean-360'])
    training_data = training_data_100.concatenate(training_data_360)

    # pronouncing_dictionary, phoneme_mapping, alphabet_size = load_cmu()

    keys = tf.constant(ALPHABET)
    char_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=keys,
            values=tf.range(1, tf.shape(keys) + 1)),
        default_value=0)

    def _preprocess_example(speech, label):
        return preprocess_example(speech, label, char_table)

    validation_data = validation_data.map(_preprocess_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    training_data = training_data.map(_preprocess_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    training_data = training_data.filter(filter_empty).shuffle(SHUFFLE_BUFFER_SIZE).padded_batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    validation_data = validation_data.padded_batch(BATCH_SIZE).filter(filter_empty).prefetch(tf.data.experimental.AUTOTUNE)

    model = create_model(len(ALPHABET))
    compile_model(model)

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
        monitor="val_loss", patience=2, restore_best_weights=True
    ))

    model.fit(
        training_data,
        validation_data=validation_data,
        epochs=100,
        callbacks=callbacks,
    )
    
    model.save("tmp/model.h5")
