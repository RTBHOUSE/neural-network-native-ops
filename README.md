# neural network native operations

Small Java lib with few neural network operations:
 - ReLU
 - linearForward
 - and simple matrix-by-vector multiplication (gemv).

Behind the scenes it uses OpenBlas native library
hence it's even order of magnitude faster than pure Java implementation.

One drawback is that full performance is achieved only on direct float buffers,
which are expensive to create so must be reused.

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

#### gemv

Results below (table and graph) shows performance ratio between native
and pure Java implementation of matrix-by-vector multiplication (gemv)

input vector size|10|20|50|75|100|150|200|300|500|1000|2000
 ---|---|---|---|---|---|---|---|---|---|---|---
**output vector size**|||||||||||
10|0.5|1.0|2.3|3.1|4.6|5.6|7.3|9.3|11.1|11.4|12.0
20|0.9|1.7|3.0|4.9|6.9|7.9|10.6|11.9|12.3|13.8|11.7
50|1.3|2.5|4.7|6.0|8.5|7.7|10.3|11.4|13.2|13.0|9.5
75|1.4|2.9|5.1|6.8|8.5|8.6|10.7|11.3|11.4|10.4|9.3
100|1.5|3.0|5.4|7.1|8.3|8.9|11.1|12.3|12.7|10.5|9.7
150|1.7|3.3|5.7|6.6|8.5|8.7|9.2|10.7|10.6|9.8|9.7
200|1.8|3.6|5.3|6.7|8.8|9.1|11.7|11.0|9.7|9.9|9.5
300|1.8|3.7|5.4|6.6|8.9|8.9|9.7|8.9|9.6|10.5|9.9
500|2.0|3.7|3.7|6.3|8.2|7.6|8.0|9.0|9.7|9.7|8.8
1000|1.8|3.9|5.4|5.6|7.4|7.2|7.1|9.0|8.8|8.6|5.6
2000|1.8|4.0|5.0|5.3|7.6|7.2|8.3|9.0|7.7|4.4|3.9

![gemv benchmarks](https://github.com/RTBHOUSE/neural-network-native-ops/raw/master/gemv_benchmarks.png)

It shows that using native code will repay when at least one dimension
is over 20. When both dimensions are between 50 and 1000 native
multiplication outperforms pure java by order of magnitude.
Best performance is seen when input vector is large and output vector is small.
Surprisingly when both dimensions are over 1000 performance gain starts to shrink.

#### linearForward

Here are same ratios for linear forward operation:

input vector size|1|10|20|50|100|200|500|1000|2000
 ---|---|---|---|---|---|---|---|---|---
**output vector size**|||||||||
1|0.1|0.1|0.2|0.4|0.8|1.4|2.6|3.7|4.8
10|0.2|0.5|0.9|2.0|4.4|7.0|10.4|11.2|11.7
20|0.4|0.8|1.6|3.0|6.6|10.5|12.0|14.8|13.8
50|0.9|1.2|2.2|3.2|8.0|10.3|12.4|13.7|9.0
100|1.4|1.5|3.1|5.2|8.3|11.3|12.9|10.2|8.9
200|2.0|1.7|3.5|5.2|8.6|11.5|9.7|10.4|9.0
500|3.0|1.9|3.7|5.4|8.4|8.4|9.4|9.8|8.7
1000|4.1|1.8|3.8|5.2|7.6|8.2|9.7|9.6|5.5
2000|3.5|1.8|3.9|4.8|7.6|8.8|8.7|5.9|3.6

![linear forward benchmarks](https://github.com/RTBHOUSE/neural-network-native-ops/raw/master/linear_forward_benchmarks.png)

#### detailed benchmarks
And finally some detailed benchmarks for preselected dimension 300x150.
You can see that using heap float buffers slows down native processing terribly.

```
# JMH 1.11.3 (released 40 days ago)
# VM version: JDK 1.8.0_45, VM 25.45-b02
# VM invoker: /opt/java/jdk1.8.0_45/jre/bin/java
# VM options: <none>
...
# Parameters: (inputSize = 300, outputSize = 150)
...
Benchmark                                (inputSize)  (outputSize)   Mode  Cnt         Score       Error  Units
NNNOBenchmark.pureJavaReLU                       300           150  thrpt    5  10536404,265 ± 75692,695  ops/s
NNNOBenchmark.nativeHeapReLU                     300           150  thrpt    5   2489921,702 ±  8764,931  ops/s
NNNOBenchmark.nativeDirectReLU                   300           150  thrpt    5  14825171,409 ± 28517,135  ops/s

NNNOBenchmark.pureJavaGemv                       300           150  thrpt    5     25487,467 ±   149,999  ops/s
NNNOBenchmark.nativeHeapGemv                     300           150  thrpt    5     27759,599 ±  3544,044  ops/s
NNNOBenchmark.nativeDirectGemv                   300           150  thrpt    5    305614,040 ±  2719,221  ops/s

NNNOBenchmark.pureJavaLinearForward              300           150  thrpt    5     25475,030 ±    91,038  ops/s
NNNOBenchmark.nativeHeapLinearForward            300           150  thrpt    5     30342,320 ±   287,873  ops/s
NNNOBenchmark.nativeDirectLinearForward          300           150  thrpt    5    287275,461 ±  1522,721  ops/s

```
#### run benchmarks

Run of all benchmarks takes more than 1h hence they are turned off by default. To run them type
```
mvn test -P Benchmarks
```
## installation

Releases are distributed on Maven central:

```xml
<dependency>
    <groupId>com.rtbhouse.model</groupId>
    <artifactId>neural-network-native-ops</artifactId>
    <version>${nnno.version}</version>
</dependency>
```
