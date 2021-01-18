import datetime
from pathlib import Path
from typing import Tuple

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from hystric.load.librispeech import load_librispeech
from hystric.load.cmu_dictionary import load_cmu
from hystric.model import compile_model, create_model, SAMPLES_PER_FRAME, SAMPLES_PER_HOP

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# mixed_precision.set_policy(mixed_precision.Policy('mixed_float16'))

SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 64
CHECKPOINT_FILEPATH = Path("tmp/checkpoint/cp-{epoch:04d}.ckpt")
CHECKPOINT_DIR = CHECKPOINT_FILEPATH.parent
PCM_16_MAX = 2**15


def preprocess_example(speech, label, pronouncing_dictionary_index, pronouncing_dictionary_values):
    return preprocess_audio(speech), preprocess_label(label, pronouncing_dictionary_index, pronouncing_dictionary_values)

def preprocess_audio(audio):
    '''Convert PCM to normalised floats and chunk audio into frames of correct size to feed to RNN'''
    return tf.signal.frame(tf.cast(audio, 'float32') / PCM_16_MAX, frame_length=SAMPLES_PER_FRAME, frame_step=SAMPLES_PER_HOP)

def preprocess_label(label: tf.Tensor, pronouncing_dictionary_index: tf.lookup.StaticHashTable, pronouncing_dictionary_values: tf.RaggedTensor):
    word_indices = pronouncing_dictionary_index.lookup(tf.strings.split(tf.strings.upper(label)))
    return tf.gather(pronouncing_dictionary_values, word_indices).merge_dims(0, 1)


def train():
    validation_data, training_data_100, training_data_360 = load_librispeech(splits=['dev-clean', 'train-clean-100', 'train-clean-360'])
    training_data = training_data_100.concatenate(training_data_360)

    pronouncing_dictionary, phoneme_mapping, alphabet_size = load_cmu()

    keys = tf.constant(list(pronouncing_dictionary.keys()))
    pronouncing_dictionary_index = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=keys,
            values=tf.range(1, tf.shape(keys) + 1)),
        default_value=0)

    pronouncing_dictionary_values = tf.ragged.constant([[]] + list(pronouncing_dictionary.values()))

    def _preprocess_example(speech, label):
        return preprocess_example(speech, label, pronouncing_dictionary_index, pronouncing_dictionary_values)

    validation_data = validation_data.map(_preprocess_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    training_data = training_data.map(_preprocess_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    training_data = training_data.shuffle(SHUFFLE_BUFFER_SIZE).padded_batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    validation_data = validation_data.padded_batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    model = create_model(alphabet_size)
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
        monitor="val_loss", patience=4, restore_best_weights=True
    ))

    model.fit(
        training_data,
        validation_data=validation_data,
        epochs=100,
        callbacks=callbacks,
    )
    
    model.save("tmp/model.h5")
