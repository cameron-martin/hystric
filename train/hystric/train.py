import datetime
from pathlib import Path
from typing import List, Tuple
import re
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from hystric.load.librispeech import load_librispeech
from hystric.load.common_voice import load_common_voice
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

def normalise_label(label):
    label = tf.strings.regex_replace(label, "-", " ")
    label = tf.strings.regex_replace(label, "[^'a-zA-Z0-9\s]", "")
    label = tf.strings.regex_replace(label, "\s+", " ")
    label = tf.strings.upper(label)
    label = tf.strings.strip(label)
    label = label + " "
    return label

def preprocess_label(label: tf.Tensor, char_table: tf.lookup.StaticHashTable):
    return char_table.lookup(tf.strings.bytes_split(normalise_label(label)))

ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ\' ')

def interleave_datasets(datasets: List[tf.data.Dataset]):
    return tf.data.Dataset.from_tensor_slices(datasets).interleave(tf.identity, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

def load_datasets():
    librispeech_validation_data, librispeech_training_data_clean_100, librispeech_training_data_clean_360, librispeech_training_data_other_500 = load_librispeech(splits=['dev-clean', 'train-clean-100', 'train-clean-360', 'train-other-500'])
    common_voice_training_data = load_common_voice(splits=['train'])
    
    training_data = interleave_datasets([librispeech_training_data_clean_100, librispeech_training_data_clean_360, librispeech_training_data_other_500, common_voice_training_data])

    return training_data, librispeech_validation_data

def train():
    training_data, validation_data = load_datasets()

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

    initial_epoch = 0
    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest is not None:
        model.load_weights(latest)
        initial_epoch = int(re.match(r'cp-([0-9]+)\.', os.path.basename(latest)).group(1))

    callbacks = []

    callbacks.append(keras.callbacks.ModelCheckpoint(
        filepath=str(CHECKPOINT_FILEPATH), save_weights_only=True
    ))

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
        initial_epoch=initial_epoch
    )
    
    model.save("tmp/model.h5")
