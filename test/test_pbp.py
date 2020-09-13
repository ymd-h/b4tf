import unittest

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

        self.assertEqual(pbp._logZ(diff2,v),
                         -0.5*tf.math.log(2.0*pi*v))



if __name__ == "__main__":
    unittest.main()
