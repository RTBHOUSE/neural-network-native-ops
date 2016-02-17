# neural network native operations

Small java lib with few neural network operations: ReLU, linearForward and simple matrix-by-vector multiplication.
Under the sceens it uses OpenBlas native library.

## Building native part

We supply precompiled library for Linux sandybridge 64 bit processor.

To run under other processor / architecture one need to compile few things:
 1. `git clone https://github.com/xianyi/OpenBLAS.git`
 2. go to OpenBlas dir and `make` (or `make USE_THREAD=0` when you don't want multithreading)
 3. `make PREFIX=~/OpenBLASlib install`
 4. go to neural-network-native-ops dir and `mvn clean compile`
 5. `java -jar ~/.m2/repository/org/bytedeco/javacpp/1.1/javacpp-1.1.jar -classpath target/classes  com.rtbhouse.model.natives.* -Xcompiler -I$HOME/OpenBLASlib/include/ -Xcompiler ~/OpenBLASlib/lib/libopenblas.a`
 6. `mv target/classes/com/rtbhouse/model/natives/*/*.so src/main/resources/com/rtbhouse/model/natives/`
 7. `mvn clean install`



