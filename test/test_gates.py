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
            BooleanReversibleGate(lambda x: (x,)),
            IdentityGate(1)
        )
        np.testing.assert_array_equal(
            BooleanReversibleGate(lambda x: [not x]).matrix_representation(),
            np.array([
                [0, 1],
                [1, 0]
            ])
        )
        swap_gate = BooleanReversibleGate(lambda x, y: (y, x))
        np.testing.assert_array_equal(
            swap_gate.matrix_representation(),
            np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ])
        )
        with self.assertRaises(ValueError):
            BooleanReversibleGate(lambda x: [False])

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
        q = QubitArray.from_bits((0, 1))
        g = TensorProductGate(IdentityGate(1), PauliX())
        self.assertEqual(
            g(q),
            QubitArray.from_bits((0, 0))
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
        toffoli = ControlledGate(3, cnot, 1, 0)
        np.testing.assert_array_equal(
            toffoli.matrix_representation(),
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0]
            ])
        )

    def test_circuit(self):
        sqrt_not = Circuit(HadamardGate(1), PhaseShiftGate(-np.pi / 2), HadamardGate(1))
        x = Circuit(sqrt_not, sqrt_not)
        np.testing.assert_array_almost_equal(
            x.matrix_representation(),
            PauliX().matrix_representation()
        )
