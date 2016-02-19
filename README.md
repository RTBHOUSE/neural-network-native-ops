# neural network native operations

Small Java lib with few neural network operations:
 - ReLU
 - linearForward
 - and simple matrix-by-vector multiplication.

Behind the scenes it uses OpenBlas native library
hence it's order of magnitude faster than pure Java implementations.

One drawback is that full performance is achieved only on direct float buffers,
which are expensive to create and not so convenient as plain Java arrays.

## building native part

We supply precompiled library for Linux sandybridge 64 bit processor
with `SSE` and `AVX` instruction set, but without `AVX2`.

To run under other processor / architecture one need to compile few things:
 1. Get [OpenBLAS](https://github.com/xianyi/OpenBLAS)
 2. Compile it

    `make`

    (or `make USE_THREAD=0` when you don't want multithreading)
 3. and install **precisely** like this

    `make PREFIX=~/OpenBLASlib install`
 4. Finally go to neural-network-native-ops dir and

    `mvn clean compile exec:exec install`

    The `exec:exec` goal will execute `javacpp` postprocessing to
    generate C++ file and finaly `g++` compiler to produce JNI lib (`.so`).

## benchmarks

Benchmarks was run on my desktop machine
```
$ cat /proc/cpuinfo
...
Intel(R) Core(TM) i5-4460  CPU @ 3.20GHz
model name	: Intel(R) Core(TM) i5-4460  CPU @ 3.20GHz
cpu MHz		: 3288.750
cache size	: 6144 KB
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm ida arat xsaveopt pln pts dtherm tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid
bogomips	: 6400.01
```
Results shows that native matrix multiplication is order of magnitude faster. ReLU is silghtly faster.

```
# JMH 1.11.3 (released 36 days ago)
# VM version: JDK 1.8.0_45, VM 25.45-b02
# VM invoker: /opt/java/jdk1.8.0_45/jre/bin/java
# VM options: <none>
# Warmup: 10 iterations, 1 s each
# Measurement: 10 iterations, 1 s each
...

Benchmark                                                           Mode  Cnt         Score        Error  Units
NeuralNetworkNativeOpsBenchmark.benchmakPureJavaReLU               thrpt   10  10278427,681 ± 116622,341  ops/s
NeuralNetworkNativeOpsBenchmark.benchmakNativeHeapReLU             thrpt   10   2256273,726 ±  23581,420  ops/s
NeuralNetworkNativeOpsBenchmark.benchmakNativeDirectReLU           thrpt   10  14205993,337 ±  83869,004  ops/s

NeuralNetworkNativeOpsBenchmark.benchmarkPureJavaGemv              thrpt   10     25156,073 ±    259,454  ops/s
NeuralNetworkNativeOpsBenchmark.benchmakNativeHeapGemv             thrpt   10     29360,148 ±    602,002  ops/s
NeuralNetworkNativeOpsBenchmark.benchmakNativeDirectGemv           thrpt   10    295473,262 ±   1709,584  ops/s

NeuralNetworkNativeOpsBenchmark.benchmakNativeHeapLinearForward    thrpt   10     29073,979 ±    778,664  ops/s
NeuralNetworkNativeOpsBenchmark.benchmakNativeDirectLinearForward  thrpt   10    266397,395 ±   2303,187  ops/s
```
