# c-backprop
A small Backpropagation implementation in C, capable of computing gradients of scalar functions.

The computational graph is Pyotrch like (i.e. computed at runtime).

## Run
Currently this implementation is **compiler dependent** and needs *gcc* (see `src/constant.c` for details).

In order to run you must install *gcc* and *make*.

- Compile `main.c` as the entry point with
    ```
    make 
    ```
    and run  with
    ```
    ./c_backprop
    ```

- Compile tests for numeric type with
    ```
    make test-numeric
    ```
    and run  with
    ```
    ./test_numeric_bin
    ```

- Compile tests for tensor type (TODO)
    
- Clean executables
    ```
    make clean
    ```


## TODO:
- [ ] Modify Makefile to compile only numeric files with argument test-numeric
- [ ] Add const to each input of functions requiring it
- [ ] Verify slicing correctness
- [ ] Add proper tensor error handling
- [ ] Add proper comments (documentation) to tensor functions
- [ ] Implement tensor_math operations
- [ ] Verify destroy function (right now I'm not sure if we are handling it correctly)
- [ ] Add tests for tensor functions