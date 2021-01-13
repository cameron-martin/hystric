import datetime
from pathlib import Path
from typing import Tuple
from pydub.audio_segment import AudioSegment

import tensorflow as tf
import numpy as np
from tensorflow import keras
import kapre
import tensorflow_datasets as tfds
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from hystric.load.librispeech import load_librispeech
from hystric.load.cmu_dictionary import load_cmu
from hystric.model import compile_model, create_model, SAMPLES_PER_FRAME, SAMPLES_PER_HOP
import pydub.generators
import pydub.playback
import random

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# mixed_precision.set_policy(mixed_precision.Policy('mixed_float16'))

SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 8
CHECKPOINT_FILEPATH = Path("tmp/checkpoint/cp-{epoch:04d}.ckpt")
CHECKPOINT_DIR = CHECKPOINT_FILEPATH.parent
PCM_16_MAX = 2**15

def generate_sin_wave(sample_rate, time_ms: int, frequency: int):
    sine_wave = pydub.generators.Sine(frequency, sample_rate=sample_rate, bit_depth=16)
    sine_segment = sine_wave.to_audio_segment(duration=time_ms) 
    return sine_segment

def generate_example():
    wave1 = pydub.generators.Sine(400, sample_rate=16000, bit_depth=16)
    wave2 = pydub.generators.Sine(1000, sample_rate=16000, bit_depth=16)


    current_wave = (wave1, 1) if random.randrange(0, 2) == 0 else (wave2, 2)
    n_segments = random.randrange(5, 20)

    label = []
    audio = AudioSegment.empty()
    
    for _ in range(0, n_segments):
        segment_length = random.uniform(200, 500)
        label.append(current_wave[1])
        audio += current_wave[0].to_audio_segment(duration=segment_length)
        current_wave = (wave1, 1) if current_wave[1] == 2 else (wave2, 2)

    return tf.constant(audio.get_array_of_samples()), tf.constant(label)

def generate_examples():
    while True:
        yield generate_example()

# def test_sin_wave():
#     audio, label = generate_example()
#     print(label)
#     play_obj = sa.play_buffer(tf.expand_dims(audio, -1).numpy().astype(np.int16), 1, 2, 16000)
#     play_obj.wait_done()

def preprocess_example(speech, label):
    return preprocess_audio(speech), label

def preprocess_audio(audio):
    '''Convert PCM to normalised floats and chunk audio into frames of correct size to feed to RNN'''
    return tf.signal.frame(tf.cast(audio, 'float32') / PCM_16_MAX, frame_length=SAMPLES_PER_FRAME, frame_step=SAMPLES_PER_HOP)

def preprocess_label(label: tf.Tensor, pronouncing_dictionary_index: tf.lookup.StaticHashTable, pronouncing_dictionary_values: tf.RaggedTensor):
    word_indices = pronouncing_dictionary_index.lookup(tf.strings.split(tf.strings.upper(label)))
    return tf.gather(pronouncing_dictionary_values, word_indices).merge_dims(0, 1)


def train():
    output_signature = (tf.TensorSpec((None,)), tf.TensorSpec((None,), dtype='int64'))
    validation_data = tf.data.Dataset.from_generator(generate_examples, output_signature=output_signature)
    training_data = tf.data.Dataset.from_generator(generate_examples, output_signature=output_signature)

    validation_data = validation_data.map(preprocess_example)
    training_data = training_data.map(preprocess_example)

    training_data = training_data.padded_batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    validation_data = validation_data.padded_batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    model = create_model(alphabet_size=2)
    compile_model(model)

    model.summary()

    callbacks = []

    log_dir = "tmp/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0))

    # callbacks.append(tf.keras.callbacks.EarlyStopping(
    #     monitor="val_loss", patience=4, restore_best_weights=True
    # ))

    model.fit(
        training_data,
        validation_data=validation_data,
        epochs=100,
        callbacks=callbacks,
        steps_per_epoch=1000,
        validation_steps=50,
    )
    
    model.save("tmp/model.h5")
