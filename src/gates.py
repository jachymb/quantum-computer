import functools
import operator
from abc import ABC, abstractmethod
from typing import Callable, Collection, Self

import matrix_properties
import bits
import numpy as np
import scipy.linalg

from _representation import _complex
from qubits import QubitArray


class QuantumGate(ABC):
    """
    Base class for various gates.
    """
    def __init__(self, size: int):
        self._size = size

    @property
    def size(self):
        return self._size

    @abstractmethod
    def matrix_representation(self) -> np.ndarray: ...

    def __call__(self, a: QubitArray) -> QubitArray:
        return QubitArray(self.matrix_representation() @ a.vector_representation)

    def __eq__(self, other: Self) -> bool:
        return np.array_equal(self.matrix_representation(), other.matrix_representation())


class IdentityGate(QuantumGate):
    """
    The no-op gate.
    """

    def matrix_representation(self) -> np.ndarray:
        return np.eye(2 ** self._size, dtype=_complex)


class TensorProductGate(QuantumGate):
    """
    Combine gates together to represent parallel computation.
    """

    def __init__(self, *gates: QuantumGate):
        self._gates = gates
        super().__init__(sum(g._size for g in gates))

    def matrix_representation(self) -> np.ndarray:
        matrices = (g.matrix_representation() for g in self._gates)
        empty = np.array([[1]], dtype=_complex)
        return functools.reduce(np.kron, matrices, empty)


class PauliX(QuantumGate):
    """1-qubit Pauli X gate, a.k.a. the NOT gate."""
    def __init__(self):
        super().__init__(1)

    def matrix_representation(self) -> np.ndarray:
        return np.array([
            [0, 1],
            [1, 0]
        ], dtype=_complex)


class HadamardGate(QuantumGate):
    def matrix_representation(self) -> np.ndarray:
        normalization = 2 ** (self._size / 2)
        return scipy.linalg.hadamard(2 ** self._size, dtype=_complex) / normalization


class BooleanReversible(QuantumGate):
    """
    Converts a function f: {False, True}ⁿ → {False, True}ⁿ to the corresponding quantum gate.
    The function must be a permutation (a.k.a. reversible).
    """

    def __init__(self, size: int, f: Callable[[Collection[bool]], Collection[bool]], validate=True):
        """
        :param size: The number of input/output bits of the boolean function.
        :param f: The boolean function.
        :param validate: Whether to check if it's a permutation. Raises a ValueError if it's not a permutation.
        """
        super().__init__(size)
        self.f = f
        if validate:
            self._validate()

    def matrix_representation(self) -> np.ndarray:
        n = 2 ** self._size
        m = np.zeros((n, n), dtype=_complex)
        for i, a in enumerate(bits.all_values(self._size)):
            m[i, bits.to_int_big_endian(self.f(a))] = 1
        return m

    def _validate(self):
        if not matrix_properties.is_permutation_matrix(self.matrix_representation()):
            raise ValueError("Function is not a permutation!")


class ControlledGate(QuantumGate):
    def __init__(self, size: int, base_gate: QuantumGate, at_qubit: int, controlled_by: int):
        super().__init__(size)
        self.base_gate = base_gate
        self.at_qubit = at_qubit
        self.controlled_by = controlled_by

        if base_gate.size != 1:
            raise NotImplementedError("Only single qubit controlled gates are supported.")

        if not 0 <= controlled_by < self._size:
            raise ValueError("Invalid base gate qubit.")

        if not 0 <= controlled_by < self._size:
            raise ValueError("Invalid number of control qubit.")

        if controlled_by == at_qubit:
            raise ValueError("Base gate cannot be at the same position as control qubit.")

    def matrix_representation(self) -> np.ndarray:
        m = TensorProductGate(
            IdentityGate(self.at_qubit),
            self.base_gate,
            IdentityGate(self._size - self.at_qubit - 1)
        ).matrix_representation()
        return np.array([
                m @ (QubitArray.from_bits(*a)).vector_representation
                if a[self.controlled_by] else
                QubitArray.from_bits(*a).vector_representation
                for i, a in enumerate(bits.all_values(self._size))
        ])


class Oracle(QuantumGate):
    """
    A gate represented by an arbitrary unitary transformation.
    """

    def __init__(self, matrix: np.ndarray, validate=True):
        self._matrix = matrix

        if validate:
            self._validate()

        super().__init__(int(np.log2(self._matrix.shape[0])))

    def _validate(self):
        n, m = self._matrix.shape
        if n != m:
            raise ValueError("Matrix is not square!")
        elif not bits.is_power_of_two(n):
            raise ValueError(f"Matrix size {n} is not a power of two!")
        elif n == 0:
            raise ValueError("Cannot initialize from empty vector.")
        elif not matrix_properties.is_unitary(self._matrix):
            raise ValueError("Matrix is not unitary!")

    def matrix_representation(self) -> np.ndarray:
        return self._matrix


class Circuit(QuantumGate):
    def __init__(self, *gates: QuantumGate):
        size = gates[0].size
        if any(g.size != size for g in gates):
            raise ValueError("Incompatible gate sizes!")
        super().__init__(size)
        self.gates = gates

    def matrix_representation(self) -> np.ndarray:
        matrices = (g.matrix_representation() for g in reversed(self.gates))
        return functools.reduce(operator.matmul, matrices)
