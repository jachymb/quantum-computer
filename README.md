This is a simple implementation of quantum circuitry emulation framework.

It is kind of similar to QisKit but less powerful, so if you want to do something serious with quantum computation use QisKit instead. I wanted to try to implement this myself as a hobby project and I didn't inspect QisKit sources and only learned about it when I was finishing this.

The code is designed as a re-usable library.

But if you've ever wondered whether the boolean function `f(x)=1` is constant and needed to compute the answer really quickly using a quantum computer,
but actually very slowly because of emulation, the famous Deutsch-Jozsa algorithm is implemented using this framework.

I guess it could be used for educational purposes, idk?
