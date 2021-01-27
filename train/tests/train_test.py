import tensorflow as tf
import numpy as np

from hystric.train import normalise_label

def test_normalise_label():
    inputs = tf.constant([
        "Cameron's GPU is lit",
        "The thing is - I know - that this NN's performance is terrible.",
        "Faustin-Archange Touadéra is elected for a second term as President of the Central African Republic."
    ])

    expected_outputs = tf.constant([
        "CAMERON'S GPU IS LIT ",
        "THE THING IS I KNOW THAT THIS NN'S PERFORMANCE IS TERRIBLE ",
        "FAUSTIN ARCHANGE TOUADRA IS ELECTED FOR A SECOND TERM AS PRESIDENT OF THE CENTRAL AFRICAN REPUBLIC "
    ])

    actual_outputs = normalise_label(inputs)

    assert np.all(actual_outputs.numpy() == expected_outputs.numpy()) 