package com.rtbhouse.model.natives;

import static com.rtbhouse.model.natives.NeuralNetworkNativeOpsTest.MAX_ERROR;
import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.ChainedOptionsBuilder;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

import com.rtbhouse.tests.Benchmark;
import com.rtbhouse.tests.ReleaseBenchmark;

public class NNNOBenchmarkTest {

    private final float[] A = new float[] { 1.1f, -1.02f, 1.003f, -1.0004f, 1.00005f, -1.000006f };
    private final float[] x = new float[] { 2, -1 };
    private final float[] y = new float[] { -1, 0, 1 };
    private float[] expectedAbyXplusY = new float[] { 2.22f, 3.0064f, 4.000106f };


    private ChainedOptionsBuilder benchmarkGenericOptions = new OptionsBuilder()
            .warmupTime(TimeValue.seconds(1))
            .warmupIterations(5)
            .measurementTime(TimeValue.seconds(1))
            .measurementIterations(10)
            .threads(1)
            .forks(5)
            .shouldFailOnError(true)
            .shouldDoGC(true);

    @Test
    public void testPureJavaGemv() {
        // when
        NNNOBenchmark.pureJavaGemv(A, x, y);
        // then
        assertArrayEquals(expectedAbyXplusY, y, MAX_ERROR);
    }

    @Test
    public void testNetlibGemv() {
        // when
        NNNOBenchmark.netlibJavaGemv(A, x, y);
        // then
        assertArrayEquals(expectedAbyXplusY, y, MAX_ERROR);
    }

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
    public void nativeVsPureJavaGemvBenchmark() throws Exception {
        Options opts = benchmarkGenericOptions
                .include("nativeDirectGemv")
                .include("pureJavaGemv")
                .build();

        new Runner(opts).run();
    }


    @Test
    @Category(Benchmark.class)
    public void nativeVsNetlibJavaGemvBenchmark() throws Exception {
        Options opts = benchmarkGenericOptions
                .include("nativeDirectGemv")
                .include("netlibJavaGemv")
                .build();

        new Runner(opts).run();
    }

    @Test
    @Category(Benchmark.class)
    public void nativeVsPureJavaLinearForwardBenchmarks() throws Exception {
        Options opts = benchmarkGenericOptions
                .include("nativeDirectLinearForward")
                .include("pureJavaLinearForward")
                .build();

        new Runner(opts).run();
    }

    @Test
    @Category(ReleaseBenchmark.class)
    public void releaseBenchmark() throws Exception {
        Options opts = benchmarkGenericOptions
                .include("nativeDirectGemv")
                .include("pureJavaGemv")
                .param("inputSize", "300")
                .param("outputSize", "150")
                .build();

        new Runner(opts).run();
    }
}
