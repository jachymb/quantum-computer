"""
The implementation of the Deuths algorithm.
"""
from typing import Callable

import numpy as np
from _representation import _complex
from gates import QuantumGate, Circuit, HadamardGate, TensorProductGate, IdentityGate
from qubits import QubitArray


class DeutschOracle(QuantumGate):
    def __init__(self, f: Callable[[bool], bool]):
        super().__init__(2)
        self.f = f

    def matrix_representation(self) -> np.ndarray:
        f = self.f
        return np.array([
            [False is f(False), False is (f(False) ^ True), 0, 0],
            [True is f(False), True is (f(False) ^ True), 0, 0],
            [0, 0, False is f(True), False is (f(True) ^ True)],
            [0, 0, True is f(True), True is (f(True) ^ True)],
        ], dtype=_complex)


def deutsh_algorithm(f):
    c = Circuit(
        HadamardGate(2),
        DeutschOracle(f),
        TensorProductGate(HadamardGate(1), IdentityGate(1))
    )
    return c(QubitArray.from_bits(0, 1)).born_rule()


if __name__ == "__main__":
    print(deutsh_algorithm(lambda x: False))
    print(deutsh_algorithm(lambda x: True))
    print(deutsh_algorithm(lambda x: x))
    print(deutsh_algorithm(lambda x: not x))
