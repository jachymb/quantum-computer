import numpy as np
import scipy.linalg
import itertools

_complex = np.complex128

ket0 = np.array((1, 0), dtype=_complex)
ket1 = np.array((0, 1), dtype=_complex)

def tensor_product(a, b):
    return np.tensordot(a, b, axes=0)

pauliX = np.array([[0, 1], [1, 0]], dtype=_complex)
pauliY = np.array([[0, -1j], [1j, 0]], dtype=_complex)
pauliZ = np.array([[1, 0], [0, -1]], dtype=_complex)

def tensor_power(a, n):
    if n == 0:
        return _complex(1.)
    else:
        return tensor_product(a, tensor_power(a, n-1))


def hadamard(n_qubits: int):
    h = _complex(1./np.sqrt(2.))
    return tensor_power([[h, h], [h, -h]], n_qubits)
    #if n_qubits == 0:
    #    return _complex(1.)
    #else:
    #    h = hadamard(n_qubits - 1)
    #    return np.array([[h, h], [h, -h]]) / np.sqrt(2)

def identity(n_qubits: int):
    return tensor_power(np.eye(2), n_qubits)

def rotX(theta):
    return scipy.linalg.expm(-0.5j * theta * pauliX)
def rotY(theta):
    return scipy.linalg.expm(-0.5j * theta * pauliY)
def rotZ(theta):
    return scipy.linalg.expm(-0.5j * theta * pauliZ)
def born_rule(quantum_state):
    return np.abs(quantum_state)**2

def projection(from_n_qubits: int, onto: set[int]):
    ...


h1 = hadamard(1)
h2 = hadamard(2)
i1 = identity(1)
i2 = identity(2)

zero1 = np.zeros((2, 2), dtype=_complex)

cNot_0_1 = np.array([[i1, zero1], [pauliX, zero1]])*2 #scipy.linalg.block_diag(i1, pauliX)
cNot_1_0 = np.array([[pauliX, zero1], [zero1, i1]])*2 #scipy.linalg.block_diag(pauliX, i1)

oracle_constant0 = i2*2
oracle_constant1 = tensor_product(i1, pauliX)*2
oracle_identity = cNot_0_1
oracle_negation = cNot_1_0 #  np.kron(i1, pauliX) @ cNot_0_1

init = tensor_product(ket0, ket1)

deutsch_circuit1 = h2 @ oracle_negation @ tensor_product(h1, i1)
#deutsch_circuit2 = np.tensordot(np.tensordot(h2, oracle_negation), tensor_product(h1, i1))

print(init.shape)

print(born_rule(np.tensordot(deutsch_circuit1, init)))