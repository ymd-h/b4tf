import unittest

class TestB4TF(unittest.TestCase):
    def test_import_b4tf(self):
        import b4tf
        pbp = b4tf.PBP([2,1])

    def test_import_PBP(self):
        from b4tf.models import PBP as _PBP
        pbp = _PBP([2,1])

    def test_relative_import(self):
        from b4tf import PBP as _PBP_
        pbp = _PBP_([2,1])


if __name__ == "__main__":
    unittest.main()
