package com.rtbhouse.model.natives;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Random;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

@State(Scope.Thread)
public class NeuralNetworkNativeOpsBenchmark {

    private static final Random RANDOM = new Random();
    private static final int N = 300;
    private static final int M = 150;

    private final FloatBuffer directA = allocateDirectFloatBufferOf(N * M);
    private final FloatBuffer directX = allocateDirectFloatBufferOf(N);
    private final FloatBuffer directY = allocateDirectFloatBufferOf(M);

    private final FloatBuffer heapA = FloatBuffer.wrap(new float[N * M]);
    private final FloatBuffer heapX = FloatBuffer.wrap(new float[N]);
    private final FloatBuffer heapY = FloatBuffer.wrap(new float[M]);

    private final float[][] primitiveA = new float[M][N];
    private final float[] primitiveX = new float[N];
    private final float[] primitiveY = new float[M];

    @Setup
    public void init() throws Exception {
        randomize(directA);
        randomize(directX);
        randomize(directY);
        randomize(heapA);
        randomize(heapX);
        randomize(heapY);
        randomize(primitiveA);
        randomize(primitiveX);
        randomize(primitiveY);
    }

    @Benchmark
    public void benchmakNativeDirectReLU() {
        NeuralNetworkNativeOps.ReLU(directX);
    }

    @Benchmark
    public void benchmakNativeHeapReLU() {
        NeuralNetworkNativeOps.ReLU(heapX);
    }

    @Benchmark
    public void benchmakPureJavaReLU() {
        pureJavaReLU(primitiveX);
    }

    @Benchmark
    public void benchmakNativeDirectGemv() {
        NeuralNetworkNativeOps.gemv(directA, directX, directY);
    }

    @Benchmark
    public void benchmakNativeHeapGemv() {
        NeuralNetworkNativeOps.gemv(heapA, heapX, heapY);
    }

    @Benchmark
    public void benchmarkPureJavaGemv() {
        pureJavaGemv(primitiveA, primitiveX, primitiveY);
    }

    @Benchmark
    public void benchmakNativeDirectLinearForward() {
        NeuralNetworkNativeOps.linearForward(directA, directY, directX, directY);
    }

    @Benchmark
    public void benchmakNativeHeapLinearForward() {
        NeuralNetworkNativeOps.linearForward(heapA, heapY, heapX, heapY);
    }

    private static FloatBuffer allocateDirectFloatBufferOf(int capacity) {
        return ByteBuffer
                .allocateDirect(capacity * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
    }

    private static void randomize(FloatBuffer f) {
        for (int i = 0; i < f.capacity(); i++) {
            f.put(RANDOM.nextFloat());
        }
    }

    private static void randomize(float[] f) {
        for (int i = 0; i < f.length; i++) {
            f[i] = RANDOM.nextFloat();
        }
    }

    public static void randomize(float[][] m) {
        for (int r = m.length; --r != -1;) {
            randomize(m[r]);
        }
    }

    private static void pureJavaGemv(float[][] A, float[] x, float[] y) {
        if (y.length != A.length || x.length != A[0].length) {
            System.out.println("incompatible matrix sizes");
            System.exit(-1);
        }

        for (int r = 0; r < A.length; r++) {
            for (int c = 0; c < x.length; c++) {
                y[r] += x[c] * A[r][c];
            }
        }
    }

    private static void pureJavaReLU(float[] x) {
        for (int c = 0; c < x.length; c++) {
            x[c] = x[c] < 0 ? 0 : x[c];
        }
    }

}
