import unittest

import numpy as np
import tensorflow as tf

import b4tf

class TestIssue3(unittest.TestCase):
    def test_call_with_array(self):
        pbp = b4tf.models.PBP([10,10,1])
        x = np.array(np.arange(100)/20,copy=False)
        pbp(x)
        pbp.predict(x)

    def test_shape1_call0(self):
        pbp = b4tf.models.PBP([2,2,1])
        x = tf.constant(0.0)
        pbp(x)
        pbp.predict(x)

    def test_shape1_call1(self):
        pbp = b4tf.models.PBP([2,2,1])
        x = tf.constant((0.0,))
        pbp(x)
        pbp.predict(x)

    def test_shape1_call2(self):
        pbp = b4tf.models.PBP([2,2,1])
        x = tf.constant([0.0,0.0])
        pbp(x)
        pbp.predict(x)

    def test_shape2_call1(self):
        pbp = b4tf.models.PBP([2,2,1],input_shape=(2,))
        x = tf.constant([0.0,0.0])
        pbp(x)
        pbp.predict(x)

    def test_shape2_call2(self):
        pbp = b4tf.models.PBP([2,2,1],input_shape=(2,))
        x = tf.constant([[0.0,0.0],
                         [0.0,0.0]])
        pbp(x)
        pbp.predict(x)

if __name__ == "__main__":
    unittest.main()
