import unittest
from deutsh import *
import numpy as np

class TestDeutsch(unittest.TestCase):
    def test_deutsch(self):
        np.testing.assert_array_almost_equal(deutsh_algorithm(lambda x: False), [1 / 2, 1 / 2, 0, 0])
        np.testing.assert_array_almost_equal(deutsh_algorithm(lambda x: True), [1 / 2, 1 / 2, 0, 0])
        np.testing.assert_array_almost_equal(deutsh_algorithm(lambda x: x), [0, 0, 1 / 2, 1 / 2])
        np.testing.assert_array_almost_equal(deutsh_algorithm(lambda x: not x), [0, 0, 1 / 2, 1 / 2])
