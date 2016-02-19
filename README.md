# neural network native operations

Small Java lib with few neural network operations:
 - ReLU
 - linearForward
 - and simple matrix-by-vector multiplication.

Behind the scenes it uses OpenBlas native library
hence it's few times faster than pure Java implementations.

## Building native part

We supply precompiled library for Linux sandybridge 64 bit processor
with `SSE` and `AVX` instruction set, but without `AVX2`.

To run under other processor / architecture one need to compile few things:
 1. Get [OpenBLAS](github.com/xianyi/OpenBLAS)
 2. Compile it

    `make`

    (or `make USE_THREAD=0` when you don't want multithreading)
 3. and install **precisely** like this

    `make PREFIX=~/OpenBLASlib install`
 4. go to neural-network-native-ops dir and

    `mvn clean compile exec:exec install`

    The `exec:exec` goal will execute `javacpp` postprocessing to
    generate C++ file and finaly `g++` compiler to produce JNI lib (`.so`).
