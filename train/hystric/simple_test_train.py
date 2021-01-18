import datetime
from pathlib import Path
from typing import List

import tensorflow as tf
from tensorflow import keras
from hystric.model import create_model, SAMPLES_PER_FRAME, SAMPLES_PER_HOP, samples_to_ms, CTCEditDistance, CTCLoss
import pydub.generators
import pydub.playback
import random

# physical_devices = tf.config.list_physical_devices("GPU")
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 8
CHECKPOINT_FILEPATH = Path("tmp/checkpoint/cp-{epoch:04d}.ckpt")
CHECKPOINT_DIR = CHECKPOINT_FILEPATH.parent
PCM_16_MAX = 2**15

def generate_example():
    waves: List[pydub.generators.SignalGenerator] = [
        pydub.generators.Sine(400, sample_rate=16000, bit_depth=16),
        pydub.generators.Sine(600, sample_rate=16000, bit_depth=16),
        pydub.generators.Sine(800, sample_rate=16000, bit_depth=16),
        pydub.generators.Sine(1000, sample_rate=16000, bit_depth=16),
    ]

    def random_wave(exclude=None):
        label = random.randrange(0, 4)
        while label == exclude:
            label = random.randrange(0, 4)
        return waves[label], label


    n_frames = random.randrange(5, 20)

    labels = []
    audio = random_wave()[0].to_audio_segment(duration=samples_to_ms((SAMPLES_PER_FRAME - SAMPLES_PER_HOP)/2))
    
    last_label = None
    for _ in range(0, n_frames):
        wave, label = random_wave(exclude=last_label)
        labels.append(label + 1)
        audio += wave.to_audio_segment(duration=samples_to_ms(SAMPLES_PER_HOP))
        last_label = label

    audio += random_wave()[0].to_audio_segment(duration=samples_to_ms((SAMPLES_PER_FRAME - SAMPLES_PER_HOP)/2))

    return tf.constant(audio.get_array_of_samples()), tf.constant(labels)

def generate_examples():
    while True:
        yield generate_example()

def preprocess_example(speech, label):
    return preprocess_audio(speech), label

def preprocess_audio(audio):
    '''Convert PCM to normalised floats and chunk audio into frames of correct size to feed to RNN'''
    return tf.signal.frame(tf.cast(audio, 'float32') / PCM_16_MAX, frame_length=SAMPLES_PER_FRAME, frame_step=SAMPLES_PER_HOP)

class SequenceCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__()
        self.cat_loss = tf.keras.losses.CategoricalCrossentropy(**kwargs)

    def call(self, y_true, y_pred):
        return tf.math.reduce_sum(self.cat_loss.call(tf.one_hot(y_true, 5), y_pred), axis=-1)

def train():
    output_signature = (tf.TensorSpec((None,)), tf.TensorSpec((None,), dtype='int64'))
    validation_data = tf.data.Dataset.from_generator(generate_examples, output_signature=output_signature)
    training_data = tf.data.Dataset.from_generator(generate_examples, output_signature=output_signature)

    validation_data = validation_data.map(preprocess_example)
    training_data = training_data.map(preprocess_example)

    # for audio, label in training_data:
    #     print(audio, label)
    #     return

    training_data = training_data.padded_batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    validation_data = validation_data.padded_batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    model = create_model(alphabet_size=4)
    model.compile(
        # optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=SequenceCrossEntropyLoss(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=CTCLoss(),
        metrics=[CTCEditDistance()]
    )

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
