import unittest
from bits import *

class TestBits(unittest.TestCase):
    def test_bits_to_int_big_endian(self):
        self.assertEqual(to_int_big_endian(()), 0)
        self.assertEqual(to_int_big_endian([True]), 1)
        self.assertEqual(to_int_big_endian([True, True]), 3)
        self.assertEqual(to_int_big_endian([True, True, False]), 6)

    def test_is_power_of_two(self):
        self.assertTrue(is_power_of_two(1))
        self.assertTrue(is_power_of_two(8))
        self.assertFalse(is_power_of_two(0))
        self.assertFalse(is_power_of_two(7))


if __name__ == '__main__':
    unittest.main()
