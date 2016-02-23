package com.rtbhouse.model.natives;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Random;

import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

@State(Scope.Thread)
public class NeuralNetworkNativeOpsBenchmark {
    private static final Random RANDOM = new Random();

    @Param({ "10", "20", "50", "75", "100", "150", "200", "300", "500", "1000", "2000" })
    private int inputVectorSize;
    @Param({ "10", "20", "50", "75", "100", "150", "200", "300", "500", "1000", "2000" })
    private int outputVectorSize;

    private FloatBuffer directMatrix;
    private FloatBuffer directInput;
    private FloatBuffer directOutput;

    private FloatBuffer heapMatrix;
    private FloatBuffer heapInput;
    private FloatBuffer heapOutput;

    private float[][] primitiveMatrix;
    private float[] primitiveInput;
    private float[] primitiveOutput;

    @Setup
    public void init() throws Exception {
        directMatrix = allocateDirectFloatBufferOf(inputVectorSize * outputVectorSize);
        directInput = allocateDirectFloatBufferOf(inputVectorSize);
        directOutput = allocateDirectFloatBufferOf(outputVectorSize);
        randomize(directMatrix);
        randomize(directInput);
        randomize(directOutput);

        heapMatrix = FloatBuffer.wrap(new float[inputVectorSize * outputVectorSize]);
        heapInput = FloatBuffer.wrap(new float[inputVectorSize]);
        heapOutput = FloatBuffer.wrap(new float[outputVectorSize]);
        randomize(heapMatrix);
        randomize(heapInput);
        randomize(heapOutput);

        primitiveMatrix = new float[outputVectorSize][inputVectorSize];
        primitiveInput = new float[inputVectorSize];
        primitiveOutput = new float[outputVectorSize];
        randomize(primitiveMatrix);
        randomize(primitiveInput);
        randomize(primitiveOutput);
    }

    @Benchmark
    public void benchmakNativeDirectReLU() {
        NeuralNetworkNativeOps.ReLU(directInput);
    }

    @Benchmark
    public void benchmakNativeHeapReLU() {
        NeuralNetworkNativeOps.ReLU(heapInput);
    }

    @Benchmark
    public void benchmakPureJavaReLU() {
        pureJavaReLU(primitiveInput);
    }

    @Benchmark
    public void benchmakNativeDirectGemv() {
        NeuralNetworkNativeOps.gemv(directMatrix, directInput, directOutput);
    }

    @Benchmark
    public void benchmakNativeHeapGemv() {
        NeuralNetworkNativeOps.gemv(heapMatrix, heapInput, heapOutput);
    }

    @Benchmark
    public void benchmarkPureJavaGemv() {
        pureJavaGemv(primitiveMatrix, primitiveInput, primitiveOutput);
    }

    @Benchmark
    public void benchmakNativeDirectLinearForward() {
        // second parameter simulates biases
        NeuralNetworkNativeOps.linearForward(directMatrix, directOutput, directInput, directOutput);
    }

    @Benchmark
    public void benchmakNativeHeapLinearForward() {
        // second parameter simulates biases
        NeuralNetworkNativeOps.linearForward(heapMatrix, heapOutput, heapInput, heapOutput);
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
