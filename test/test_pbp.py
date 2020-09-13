import unittest

import tensorflow as tf
from b4tf.models import PBP, PBPLayer, PBPReLULayer


class TestPBP(unittest.TestCase):
    def test_init(self):
        pbp = PBP([2,1])

        self.assertEqual(len(pbp.layers),2)
        self.assertIsInstance(pbp.layers[0],PBPReLULayer)
        self.assertIsInstance(pbp.layers[1],PBPLayer)
        self.assertEqual(pbp.input_shape,(1,))


    def test_logZ(self):
        pbp = PBP([2,1])

        diff2 = tf.constant(0.0)
        v = tf.constant(1.0)
        pi = tf.math.atan(1.0) * 4

        self.assertTrue(tf.reduce_all(pbp._logZ(diff2,v) == -0.5*tf.math.log(2.0*pi*v)))


class TestPBPLayer(unittest.TestCase):
    def test_init(self):
        layer = PBPLayer(10)
        self.assertEqual(layer.units,10)

    def test_build(self):
        layer = PBPLayer(5)
        layer.build(3)
        self.assertTrue(layer.built)

        v = layer.trainable_weights

        self.assertTrue(tf.reduce_all(layer.kernel_m == v[0]))
        self.assertTrue(tf.reduce_all(layer.kernel_v == v[1]))
        self.assertTrue(tf.reduce_all(layer.bias_m   == v[2]))
        self.assertTrue(tf.reduce_all(layer.bias_v   == v[3]))


    def test_call(self):
        layer = PBPLayer(5)
        layer.build(3)

        layer(tf.constant([[1.0,2.0,3.0],
                           [2.0,3.0,4.0]]))

if __name__ == "__main__":
    unittest.main()
