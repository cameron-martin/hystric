from pathlib import Path
import tensorflow as tf
import pyaudio

from hystric.model import SAMPLE_RATE, create_model, SAMPLES_PER_FRAME, SAMPLES_PER_HOP
from hystric.preprocessing import pcm16_to_float32


CHECKPOINT_FILEPATH = Path("tmp/checkpoint/cp-{epoch:04d}.ckpt")
CHECKPOINT_DIR = CHECKPOINT_FILEPATH.parent

ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ\' ')

def test():
    model = create_model(len(ALPHABET), stateful=True, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(CHECKPOINT_DIR))
    model.summary()

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                frames_per_buffer=SAMPLES_PER_HOP,
                input=True)
    try:
        buffer = bytearray()
        last_char_index = None
        while True:
            data = stream.read(SAMPLES_PER_HOP)
            buffer.extend(data)
            buffer = buffer[-(SAMPLES_PER_FRAME * 2):]
            if len(buffer) >= (SAMPLES_PER_FRAME * 2):
                tensor = tf.io.decode_raw(bytes(buffer), out_type='int16')
                tensor = pcm16_to_float32(tensor)
                tensor = tf.expand_dims(tensor, axis=0)
                tensor = tf.expand_dims(tensor, axis=0)
                output = model(tensor)
                char_index = tf.math.argmax(output, axis=2)[0, 0].numpy()
                if last_char_index is None or last_char_index != char_index:
                    if char_index != 0:
                        char = ALPHABET[char_index - 1]
                        print(char, end='', flush=True)
                last_char_index = char_index
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
