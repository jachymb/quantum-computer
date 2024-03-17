This is a simple implementation of quantum circuitry emulation framework.

It is kind of similar to QisKit but less powerful, so if you want to do something serious with quantum computation use QisKit instead. I wanted to try to implement this myself as a hobby project and I didn't inspect QisKit sources and only learned about it when I was finishing this.

The code is designed as a re-usable and extensible OOP/lazy-evaluation library.

But if you've ever wondered whether the black-box boolean function $f: \lbrace 0,1\rbrace  \to \lbrace 0, 1\rbrace$ is constant and needed to compute the answer really quickly using a quantum computer,
but actually very slowly because of emulation!? The famous Deutsch-Jozsa algorithm is implemented using this framework for you to solve this problem.

Cool video explaing the homemade hardware implementation for this: https://www.youtube.com/watch?v=tHfGucHtLqo

I guess it could be used for educational purposes, idk?

```python
from gates import BooleanReversibleGate, Circuit, HadamardGate, TensorProductGate, IdentityGate
from qubits import QubitArray
from typing import Callable, Collection


class DeutschOracle(BooleanReversibleGate):
    def __init__(self, f: Callable[[bool], bool]):
        super().__init__(2, lambda x: (x[0], x[1] ^ f(x[0])))


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
    print(deutsh_algorithm(lambda x: False))
    print(deutsh_algorithm(lambda x: True))
    print(deutsh_algorithm(lambda x: x))
    print(deutsh_algorithm(lambda x: not x))
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