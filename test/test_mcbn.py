import unittest

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from b4tf.models import MCBN


class TestMCBN(unittest.TestCase):
    def test_raise_init(self):
        """
        Raise ValueError without having tf.keras.layers.BatchNormalization
        """

        with self.assertRaises(ValueError):
            MCBN(Sequential([Dense(5,input_shape=(2,))]), 0.05)



if __name__ == "__main__":
    unittest.main()
