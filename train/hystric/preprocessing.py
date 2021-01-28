import tensorflow as tf
import tensorflow_io as tfio

from hystric.model import SAMPLES_PER_FRAME, SAMPLES_PER_HOP, SAMPLE_RATE

PCM_16_MAX = 2**15
ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ\' ')

keys = tf.constant(ALPHABET)
char_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=keys,
        values=tf.range(1, tf.shape(keys) + 1)),
    default_value=0)

def pcm16_to_float32(audio):
    '''Convert 16 bit PCM to floats normalised from -1 to 1'''
    return tf.cast(audio, 'float32') / PCM_16_MAX

def filter_empty(speech, label):
    '''Filter empty labels. This is necessary to avoid an infinite relative edit distance'''
    return tf.not_equal(tf.size(label), 0)

def preprocess_example(speech, label):
    return preprocess_audio(speech), preprocess_label(label)

def preprocess_audio(audio):
    '''Convert PCM to normalised floats and chunk audio into frames of correct size to feed to RNN'''
    return tf.signal.frame(pcm16_to_float32(audio), frame_length=SAMPLES_PER_FRAME, frame_step=SAMPLES_PER_HOP)

def normalise_label(label):
    label = tf.strings.regex_replace(label, "-", " ")
    label = tf.strings.regex_replace(label, "[^'a-zA-Z0-9\\s]", "")
    label = tf.strings.regex_replace(label, "\\s+", " ")
    label = tf.strings.upper(label)
    label = tf.strings.strip(label)
    label = label + " "
    return label

def preprocess_label(label: tf.Tensor):
    return char_table.lookup(tf.strings.bytes_split(normalise_label(label)))

def preprocess_dataset(dataset: tf.data.Dataset):
    return dataset.map(preprocess_example).filter(filter_empty)

def resample_48khz(audio, label):
    return tfio.audio.resample(audio, rate_in=48000, rate_out=SAMPLE_RATE), label