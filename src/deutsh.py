"""
The implementation of the Deuths algorithm.
"""
from typing import Callable, Collection

from gates import Circuit, HadamardGate, TensorProductGate, IdentityGate, BooleanReversibleGate
from qubits import QubitArray


class DeutschOracle(BooleanReversibleGate):
    def __init__(self, f: Callable[[bool], bool]):
        super().__init__(lambda x, y: (x, y ^ f(x)))


def deutsh_algorithm(f) -> Collection[float]:
    c = Circuit(
        HadamardGate(2),
        DeutschOracle(f),
        TensorProductGate(HadamardGate(1), IdentityGate(1))
    )
    return c(QubitArray.from_bits((0, 1))).measure()
