from tensorflow import keras
import tensorflow as tf
import kapre

SAMPLE_RATE = 16000

def ms_to_samples(ms: int):
    return int((SAMPLE_RATE/1000) * ms)

def samples_to_ms(samples: int):
    return samples / (SAMPLE_RATE/1000)

MFCC_WIDTH=ms_to_samples(25)
MFCC_HOP=ms_to_samples(10)
FRAME_WIDTH=5
FRAME_HOP=3

SAMPLES_PER_FRAME = (FRAME_WIDTH - 1) * MFCC_HOP + MFCC_WIDTH
SAMPLES_PER_HOP = MFCC_HOP * FRAME_HOP

UNITS=128
LSTM_LAYERS=5


class CTCLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.nn.ctc_loss(labels=y_true, logits=y_pred, label_length=tf.math.count_nonzero(y_true, -1), logit_length=tf.repeat(tf.shape(y_pred)[-2], tf.shape(y_pred)[0]), logits_time_major=False)

def CTCEditDistance():
    def ctc_edit_distance(y_true, y_pred):
        sequence_length = tf.repeat(tf.shape(y_pred)[1], tf.shape(y_pred)[0])
        # Transform blank_index from 0 to num_classes - 1 to make up for failure in API. See https://github.com/tensorflow/tensorflow/issues/42993
        y_pred_shifted = tf.roll(y_pred, shift=-1, axis=2)
        decoded, log_probability = tf.nn.ctc_greedy_decoder(tf.transpose(y_pred_shifted, (1, 0, 2)), sequence_length=sequence_length)
        decoded = decoded[0]
        # This undoes the above shift
        num_classes = tf.shape(y_pred)[2]
        decoded = tf.sparse.map_values(lambda value: tf.math.floormod(value + 1, tf.cast(num_classes, 'int64')), decoded)
        # TODO: Work out why this gets cast to a float
        y_true_sparse = tf.sparse.from_dense(tf.cast(y_true, 'int64'))
        is_nonzero = tf.not_equal(y_true_sparse.values, 0)
        y_true_sparse = tf.sparse.retain(y_true_sparse, is_nonzero)
        return tf.edit_distance(decoded, y_true_sparse)
    return ctc_edit_distance


def create_model(alphabet_size: int):
    input = keras.Input(shape=(None, SAMPLES_PER_FRAME))

    n_fft=2048

    x = keras.layers.Lambda(lambda input: tf.signal.stft(
        input,
        fft_length=n_fft,
        frame_length=MFCC_WIDTH,
        frame_step=MFCC_HOP), name="stft")(input)
    x = kapre.Magnitude()(x)
    x = keras.layers.Lambda(lambda input: tf.matmul(input, tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40,
        num_spectrogram_bins=n_fft // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=200,
        upper_edge_hertz=4000)), name="linear_to_mel")(x)
    x = keras.layers.Lambda(lambda input: tf.math.log(input + 1e-6), name="magitude_to_db")(x)
    x = keras.layers.Lambda(lambda input: tf.signal.mfccs_from_log_mel_spectrograms(input), name="mfcc")(x)
    # kapre.composed.get_melspectrogram_layer()
    # x = tf.keras.layers.AveragePooling2D(pool_size=(4, 3), data_format='channels_last')(x)
    x = keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    x = keras.layers.Dense(UNITS, activation='tanh')(x)
    for _ in range(LSTM_LAYERS):
        x = keras.layers.LSTM(UNITS, return_sequences=True)(x)
    x = keras.layers.Dense(alphabet_size + 1)(x)
    return keras.Model(input, x)

def compile_model(model: keras.Model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=CTCLoss(), metrics=[CTCEditDistance()],
    )