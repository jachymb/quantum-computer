import numpy as np


def is_permutation_matrix(m: np.ndarray) -> bool:
    return (
            m.ndim == 2
            and m.shape[0] == m.shape[1]
            and (m.sum(axis=0) == 1).all()
            and (m.sum(axis=1) == 1).all()
            and ((m == 1) | (m == 0)).all()
    )


def is_unitary(m: np.ndarray) -> bool:
    a = m @ m.conj().T
    b = m.conj().T @ m
    i = np.eye(*m.shape)
    return np.array_equal(a, b) and np.array_equal(a, i)


def is_normalized(v: np.ndarray) -> bool:
    return np.isclose(np.linalg.norm(v), 1.)
