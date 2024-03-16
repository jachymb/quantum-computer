"""
The implementation of the Deuths algorithm.
"""
from typing import Callable

import numpy as np
from _representation import _complex
from gates import QuantumGate, Circuit, HadamardGate, TensorProductGate, IdentityGate, BooleanReversibleGate
from qubits import QubitArray


class DeutschOracle(BooleanReversibleGate):
    def __init__(self, f: Callable[[bool], bool]):
        super().__init__(2, lambda x: (x[0], x[1] is not f(x[0])))


def deutsh_algorithm(f):
    c = Circuit(
        HadamardGate(2),
        DeutschOracle(f),
        TensorProductGate(HadamardGate(1), IdentityGate(1))
    )
    return c(QubitArray.from_bits(0, 1)).born_rule()
