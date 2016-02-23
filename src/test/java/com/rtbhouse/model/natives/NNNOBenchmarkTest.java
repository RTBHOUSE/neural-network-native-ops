package com.rtbhouse.model.natives;

import com.rtbhouse.tests.Benchmark;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.ChainedOptionsBuilder;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

public class NNNOBenchmarkTest {

    private ChainedOptionsBuilder benchmarkGenericOptions = new OptionsBuilder()
            .warmupTime(TimeValue.seconds(1))
            .warmupIterations(3)
            .measurementTime(TimeValue.seconds(1))
            .measurementIterations(5)
            .threads(1)
            .forks(1)
            .shouldFailOnError(true)
            .shouldDoGC(true);

    @Test
    @Category(Benchmark.class)
    public void allBenchmarksFor300x150() throws Exception {
        Options opts = benchmarkGenericOptions
                .param("inputSize", "300")
                .param("outputSize", "150")
                .build();

        new Runner(opts).run();
    }

    @Test
    @Category(Benchmark.class)
    public void nativeAndPureJavaGemvBenchmarks() throws Exception {
        Options opts = benchmarkGenericOptions
                .include("Gemv")
                .exclude("Heap")
                .build();

        new Runner(opts).run();
    }

    @Test
    @Category(Benchmark.class)
    public void nativeAndPureJavaLinearForwardBenchmarks() throws Exception {
        Options opts = benchmarkGenericOptions
                .include("LinearForward")
                .exclude("Heap")
                .build();

        new Runner(opts).run();
    }
}
