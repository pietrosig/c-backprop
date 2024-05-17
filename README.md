# c-backprop
A small Backpropagation implementation in C.
It is capable of computing gradients for scalar functions.
The computational graph is Pyotrch like (i.e. computed at runtime).

## Run
Currently this implementation is **compiler dependent** and needs *gcc* (see `src/constant.c` for details).

In order to run:

1. Install *gcc*
2. Run `chmod +x ./compile.sh` to add execute permission to the file
3. Run `./compile.sh`
4. Run `./c_backprop`
