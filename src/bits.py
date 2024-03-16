import itertools
import operator
from typing import Collection, Iterator, Tuple
import functools


def to_int_big_endian(bits: Collection[bool]) -> int:
    n = len(bits) - 1
    powers = (bit * 2**(n-i) for i, bit in enumerate(bits))
    return functools.reduce(operator.add, powers, 0)


def is_power_of_two(n: int) -> bool:
    return (n & (n - 1) == 0) and n > 0


def all_values(n: int) -> Iterator[Tuple[bool, ...]]:
    return itertools.product((False, True), repeat=n)
