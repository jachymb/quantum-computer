import functools
from typing import Self, Any
import numpy as np
import numpy.typing as npt

import bits
import matrix_properties
from _representation import _complex


class QubitArray:
    def __init__(self, vector_representation: npt.ArrayLike, validate=True):
        """
        Initializes the qubit array from a representation as a non-zero vector of complex numbers.
        The length of the representation must be a power of two.
        The vector must have unit length.
        Checking these properties may be disabled by setting the `validate` flag to `False`.
        """
        self._vector_representation = np.array(vector_representation, dtype=_complex)

        if validate:
            self._validate()

        self._n = int(np.log2(len(vector_representation)))


    def _validate(self):
        n = len(self._vector_representation)

        if not bits.is_power_of_two(n):
            raise ValueError(f"Representation array length {n} is not a power of two!")
        elif not matrix_properties.is_normalized(self._vector_representation):
            raise ValueError(f"Vector {self._vector_representation} is not normalized. Norm = {np.linalg.norm(self._vector_representation)}")

    @property
    def n(self) -> int:
        """The number of qubits in the array. (That is log2 of the size of the representation.)"""
        return self._n

    @property
    def vector_representation(self) -> np.ndarray:
        """See the internal representation as the vector in the corresponding Hilbert space."""
        return self._vector_representation

    @classmethod
    def from_tensor_product(cls, *qubit_arrays: Self) -> Self:
        """
        Construct a QubitArray as a tensor product of and arbitrary number of arrays.
        """
        return functools.reduce(cls.tensor_product, qubit_arrays, cls.empty)

    def tensor_product(self, other: Self) -> Self:
        """
        Concatenate with another QubitArray.
        This corresponds to the tensor product of the corresponding representations.
        """
        new = self.__new__(self.__class__)
        new._vector_representation = np.kron(self._vector_representation, other._vector_representation)
        new._n = self._n + other._n
        return new

    @classmethod
    @property
    def empty(cls) -> Self:
        """
        Return an empty array of qubits. This represents no-information state.
        """
        return cls((1,), False)

    @classmethod
    def from_bit(cls, bit: Any) -> Self:
        """
        Construct a 1-qubit array from a classical bit.
        The bit is interpreted as boolean, but can be any type.
        """
        return cls((0, 1), False) if bit else cls((1, 0), False)

    @classmethod
    def from_bits(cls, *bits: Any) -> Self:
        """
        Construct a qubit array from an array of classical bits.
        The bits are interpreted as booleans, but can be any type.
        """
        return cls.from_tensor_product(*map(cls.from_bit, bits))

    def born_rule(self):
        return np.abs(self._vector_representation)**2


    def __eq__(self, other: Self) -> bool:
        return np.array_equal(self._vector_representation, other._vector_representation)


ket0 = QubitArray.from_bit(0)
ket1 = QubitArray.from_bit(1)
ket00 = QubitArray.from_bits(0, 0)
ket01 = QubitArray.from_bits(0, 1)
ket10 = QubitArray.from_bits(1, 0)
ket11 = QubitArray.from_bits(1, 1)

