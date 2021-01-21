import tensorflow as tf

PCM_16_MAX = 2**15

def pcm16_to_float32(audio):
    '''Convert 16 bit PCM to floats normalised from -1 to 1'''
    return tf.cast(audio, 'float32') / PCM_16_MAX