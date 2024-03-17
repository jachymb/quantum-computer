This is a simple implementation of quantum circuitry emulation framework.

Have you ever wondered whether the black-box boolean function $f: \lbrace 0,1\rbrace  \to \lbrace 0, 1\rbrace$ is constant and needed to compute the answer really quickly using a quantum computer,
but actually very slowly because of emulation!? The infamously useless Deutsch-Jozsa algorithm is implemented using this framework for you to solve this problem!

This framework is kind of similar to QisKit but less powerful, so if you want to do something serious with quantum computation use QisKit instead. I wanted to try to implement this myself as a hobby project and I didn't inspect QisKit sources and only learned about it when I was finishing this.

The code is designed as a re-usable and extensible OOP/lazy-evaluation library.

I guess it could be used for educational purposes, idk?

The Deutsch-Jozsa quantum circuit:
![Circuit diagram](https://upload.wikimedia.org/wikipedia/commons/b/b5/Deutsch-Jozsa-algorithm-quantum-circuit.png)

That can be implemented (here for simplicity $n=1$ but can be easily generalized) as the following code: 
```python
from gates import BooleanReversibleGate, HadamardGate, TensorProductGate, IdentityGate, Circuit
from qubits import QubitArray
from typing import Callable, Collection


class DeutschOracle(BooleanReversibleGate):
    def __init__(self, f: Callable[[bool], bool]):
        super().__init__(lambda x, y: (x, y ^ f(x)))


def deutsh_algorithm(f: Callable[[bool], bool]) -> Collection[float]:
    circuit = Circuit(  # lazy declaration
        HadamardGate(2),
        DeutschOracle(f),
        TensorProductGate(HadamardGate(1), IdentityGate(1))
    )
    input_qubits = QubitArray.from_bits([0, 1])

    final_pure_state = circuit(input_qubits)  # Actual emulation happens here
    return final_pure_state.measure()  # Get probabilities of observations using Born rule


if __name__ == "__main__":
    print(deutsh_algorithm(lambda x: False))  # constant
    print(deutsh_algorithm(lambda x: True))   # constant
    print(deutsh_algorithm(lambda x: x))      # balanced
    print(deutsh_algorithm(lambda x: not x))  # balanced
```

The quantum algorithm will differentiate the balanced and the constant boolean functions using a single pass, something that is impossible on a classical computer.

```
[0.5, 0.5,    0,   0]
[0.5, 0.5,    0,   0]
[  0,   0,  0.5, 0.5]
[  0,   0,  0.5, 0.5]
```
These probabilities correspond to the observations of $|00\rangle, |01\rangle, |10\rangle$ and $|11\rangle$ respectively. 
Only the first bit is relevant. 

Cool video explaing the homemade hardware implementation for this: https://www.youtube.com/watch?v=tHfGucHtLqo

The gates API should be flexible enough to build various gates from the primitives. For example:
```python
from gates import Circuit, HadamardGate, ControlledGate, PauliX, BooleanReversibleGate, \
    PhaseShiftGate, Oracle 
from math import pi

cnot = ControlledGate(2, PauliX(), at_qubit=1, controlled_by=0)
toffoli = ControlledGate(3, cnot, at_qubit=1, controlled_by=0)
swap = BooleanReversibleGate(lambda x, y: (y, x))
sqrt_not = Circuit(HadamardGate(1), PhaseShiftGate(-pi/2), HadamardGate(1))
sqrt_swap = Oracle([
    [1, 0,        0,        0],
    [0, 0.5+0.5j, 0.5-0.5j, 0],
    [0, 0.5-0.5j, 0.5+0.5j, 0],
    [0, 0,        0,        1]
])
```
