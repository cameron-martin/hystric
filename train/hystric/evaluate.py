from pathlib import Path
from typing import List

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from hystric.load.librispeech import load_librispeech
from hystric.load.common_voice import load_common_voice
from hystric.model import CTCEditDistance, CTCLoss, create_model, SAMPLES_PER_FRAME, SAMPLES_PER_HOP
from hystric.preprocessing import ALPHABET, preprocess_dataset, resample_48khz


BATCH_SIZE = 32
CHECKPOINT_FILEPATH = Path("tmp/checkpoint/cp.ckpt")
CHECKPOINT_DIR = CHECKPOINT_FILEPATH.parent


def load_datasets():
    librispeech_validation_data_clean, librispeech_validation_data_other = load_librispeech(splits=['dev-clean', 'dev-other'])
    common_voice_validation_data, = load_common_voice(splits=['dev'])

    return {
        'librispeech_clean': librispeech_validation_data_clean.flat_map(tf.identity),
        'librispeech_other': librispeech_validation_data_other.flat_map(tf.identity),
        'common_voice': common_voice_validation_data.flat_map(tf.identity).map(resample_48khz),
    }

def evaluate():
    datasets = load_datasets()

    model = create_model(len(ALPHABET))

    model.compile(loss=CTCLoss(), metrics=[CTCEditDistance(), CTCEditDistance(beams=128)])

    model.summary()

    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest is not None:
        model.load_weights(latest)

    for name, dataset in datasets.items():
        processed_dataset = dataset.apply(preprocess_dataset).padded_batch(BATCH_SIZE, padded_shapes=([None, SAMPLES_PER_FRAME], [None])).prefetch(tf.data.AUTOTUNE)
        results = model.evaluate(x=processed_dataset)
        print(name)
        print(results)
