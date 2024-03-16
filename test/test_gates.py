import unittest
from gates import *


class TestQubit(unittest.TestCase):
    def test_identity(self):
        np.testing.assert_array_equal(
            IdentityGate(2).matrix_representation(),
            np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])
        )

    def test_hadamard(self):
        np.testing.assert_array_equal(
            HadamardGate(1).matrix_representation(),
            np.array([
                [1, 1],
                [1, -1],
            ]) / np.sqrt(2)
        )
        np.testing.assert_array_equal(
            HadamardGate(2).matrix_representation(),
            np.array([
                [1,  1,  1,  1],
                [1, -1,  1, -1],
                [1,  1, -1, -1],
                [1, -1, -1,  1]]
            ) / 2
        )

    def test_boolean_reversible(self):
        self.assertEqual(
            BooleanReversible(1, lambda x: x),
            IdentityGate(1)
        )
        np.testing.assert_array_equal(
            BooleanReversible(1, lambda x: [not x[0]]).matrix_representation(),
            np.array([
                [0, 1],
                [1, 0]
            ])
        )
        with self.assertRaises(ValueError):
            BooleanReversible(1, lambda x: [False])

    def test_tensor_product_gate(self):
        np.testing.assert_array_equal(
            TensorProductGate(IdentityGate(1), HadamardGate(1)).matrix_representation(),
            np.array([
                [1,  1,  0,  0],
                [1, -1,  0,  0],
                [0,  0,  1,  1],
                [0,  0,  1, -1]
            ]) / np.sqrt(2)
        )

    def test_call(self):
        q = QubitArray.from_bits(0, 1)
        g = TensorProductGate(IdentityGate(1), PauliX())
        self.assertEqual(
            g(q),
            QubitArray.from_bits(0, 0)
        )

    def test_controlled_gate(self):
        cnot = ControlledGate(2, PauliX(), 1, 0)
        np.testing.assert_array_equal(
            cnot.matrix_representation(),
            np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ])
        )