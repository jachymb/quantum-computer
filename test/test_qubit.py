import unittest

from qubits import *


class TestQubit(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(ValueError):
            QubitArray([1, 2, 3])
        with self.assertRaises(ValueError):
            QubitArray([])
        with self.assertRaises(ValueError):
            QubitArray([0, 0])
        with self.assertRaises(ValueError):
            QubitArray([2, 0])

    def test_from_bit(self):
        self.assertEqual(QubitArray.from_bit(False), QubitArray.from_bit(0))
        self.assertEqual(QubitArray.from_bit(True), QubitArray.from_bit(1))

    def test_from_bits(self):
        self.assertEqual(QubitArray.from_bits((0, 0)), QubitArray([1, 0, 0, 0]))

        self.assertEqual(QubitArray.from_bits((0, 0)), QubitArray([1, 0, 0, 0]))
        self.assertEqual(QubitArray.from_bits((0, 1)), QubitArray([0, 1, 0, 0]))
        self.assertEqual(QubitArray.from_bits((1, 0)), QubitArray([0, 0, 1, 0]))
        self.assertEqual(QubitArray.from_bits((1, 1)), QubitArray([0, 0, 0, 1]))

    def test_tensor_product(self):
        self.assertEqual(
            QubitArray.from_bits((0, 1)).tensor_product(QubitArray.from_bits((1, 0))),
            QubitArray.from_bits((0, 1, 1, 0))
        )
        self.assertEqual(
            QubitArray.from_bits((0, 1)).tensor_product(QubitArray.from_bits((1, 0, 0))).n,
            5
        )

    def test_n(self):
        self.assertEqual(QubitArray.from_bits((0, 1, 0)).n, 3)
        self.assertEqual(QubitArray([0, 1, 0, 0]).n, 2)

