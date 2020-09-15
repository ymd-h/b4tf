import unittest

import numpy as np
import tensorflow as tf

import b4tf

class TestIssue3(unittest.TestCase):
    def test_call_with_array(self):
        pbp = b4tf.models.PBP([10,10,1])
        x = np.array(np.arange(100)/20,copy=False)

        pbp.predict(x)


if __name__ == "__main__":
    unittest.main()
