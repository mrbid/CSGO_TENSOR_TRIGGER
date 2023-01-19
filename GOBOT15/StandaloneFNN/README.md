Three different implementations of the FNN.

- [aim1.c](aim1.c) - Column major *(fastest)*
- [aim2.c](aim2.c) - Row major with branching *(slowest)*
- [aim3.c](aim3.c) - Row major without branching or bias *(almost as fast as Column major)*

Apparently Column major is better for cache prediction.
