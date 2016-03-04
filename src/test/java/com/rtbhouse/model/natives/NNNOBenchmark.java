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

import com.github.fommil.netlib.BLAS;
import com.github.fommil.netlib.NativeSystemBLAS;

@State(Scope.Thread)
public class NNNOBenchmark {
    private static final Random RANDOM = new Random();
    static final BLAS INSTANCE = NativeSystemBLAS.getInstance();

    // uncomment this to run tests for all this presets
    @Param({ "1", "10", "20", "50", "100", "200", "500", "1000", "2000" })
    private int inputSize;
    @Param({ "1", "10", "20", "50", "100", "200", "500", "1000", "2000" })
    private int outputSize;

    private FloatBuffer directMatrix;
    private FloatBuffer directInput;
    private FloatBuffer directOutput;

    private FloatBuffer heapMatrix;
    private FloatBuffer heapInput;
    private FloatBuffer heapOutput;

    private float[] primitiveMatrix;
    private float[] primitiveInput;
    private float[] primitiveOutput;

    @Setup
    public void init() throws Exception {
        directMatrix = allocateDirectFloatBufferOf(inputSize * outputSize);
        directInput = allocateDirectFloatBufferOf(inputSize);
        directOutput = allocateDirectFloatBufferOf(outputSize);
        randomize(directMatrix);
        randomize(directInput);
        randomize(directOutput);

        heapMatrix = FloatBuffer.wrap(new float[inputSize * outputSize]);
        heapInput = FloatBuffer.wrap(new float[inputSize]);
        heapOutput = FloatBuffer.wrap(new float[outputSize]);
        randomize(heapMatrix);
        randomize(heapInput);
        randomize(heapOutput);

        primitiveMatrix = new float[outputSize * inputSize];
        primitiveInput = new float[inputSize];
        primitiveOutput = new float[outputSize];
        randomize(primitiveMatrix);
        randomize(primitiveInput);
        randomize(primitiveOutput);
    }

    @Benchmark
    public void nativeDirectReLU() {
        NeuralNetworkNativeOps.ReLU(directInput);
    }

    @Benchmark
    public void nativeHeapReLU() {
        NeuralNetworkNativeOps.ReLU(heapInput);
    }

    @Benchmark
    public void pureJavaReLU() {
        pureJavaReLU(primitiveInput);
    }

    @Benchmark
    public void nativeDirectGemv() {
        NeuralNetworkNativeOps.gemv(directMatrix, directInput, directOutput);
    }

    @Benchmark
    public void nativeHeapGemv() {
        NeuralNetworkNativeOps.gemv(heapMatrix, heapInput, heapOutput);
    }

    @Benchmark
    public void pureJavaGemv() {
        pureJavaGemv(primitiveMatrix, primitiveInput, primitiveOutput);
    }

    @Benchmark
    public void netlibJavaGemv() {
        netlibJavaGemv(primitiveMatrix, primitiveInput, primitiveOutput);
    }

    @Benchmark
    public void nativeDirectLinearForward() {
        // second parameter simulates biases
        NeuralNetworkNativeOps.linearForward(directMatrix, directOutput, directInput, directOutput);
    }

    @Benchmark
    public void nativeHeapLinearForward() {
        // second parameter simulates biases
        NeuralNetworkNativeOps.linearForward(heapMatrix, heapOutput, heapInput, heapOutput);
    }

    @Benchmark
    public void pureJavaLinearForward() {
        // second parameter simulates biases
        pureJavaLinearForward(primitiveMatrix, primitiveOutput, primitiveInput, primitiveOutput);
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

    private static void pureJavaReLU(float[] x) {
        for (int c = 0; c < x.length; c++) {
            x[c] = x[c] < 0 ? 0 : x[c];
        }
    }

    static void pureJavaGemv(float[] A, float[] x, float[] y) {
        if (y.length * x.length != A.length) {
            System.out.println("incompatible matrix sizes");
            System.exit(-1);
        }

        int i = 0;
        for (int r = 0; r < y.length; r++) {
            for (int c = 0; c < x.length; c++) {
                y[r] += x[c] * A[i++];
            }
        }
    }

    static void netlibJavaGemv(float[] matrix, float[] vector, float[] inOut) {
        INSTANCE.sgemv("T", vector.length, inOut.length, 1.0f, matrix, vector.length, vector, 1, 1.0f, inOut, 1);
    }

    private static void pureJavaLinearForward(float[] weights, float[] biases, float[] input, float[] output) {
        if (biases.length != output.length) {
            System.out.println("incompatible biases and output");
            System.exit(-1);
        }
        System.arraycopy(biases, 0, output, 0, biases.length);
        pureJavaGemv(weights, input, output);
    }

}
