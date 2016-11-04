package com.rtbhouse.model.natives;

import java.nio.FloatBuffer;

import org.bytedeco.javacpp.annotation.Platform;

import com.github.fommil.jni.JniLoader;

/**
 * <p>
 * Utility class with few operations useful when dealing with neural networks. Operations are implemented on native side
 * with BLAS behind the scenes.
 * </p>
 * Supports only single precision floating point numbers. Both heap and direct float buffers are supported but
 * an order of magnitude performance boost is achieved when using direct buffers.
 *
 * @author Piotr Chromiec
 */
@Platform(include = "NeuralNetworkNativeOps.h", compiler = "fastfpu")
public final class NeuralNetworkNativeOps {

    public static int TRANSPOSE = 0;
    public static int NO_TRANSPOSE = 1;

    static {
        JniLoader.load("com/rtbhouse/model/natives/libjniNeuralNetworkNativeOps.so");
    }

    private NeuralNetworkNativeOps() {
    }

    /**
     * <p>
     * Rectified linear unit (ReLU) function. Performs the operation in-place.
     * </p>
     * <p>
     * All input vector elements are transformed with function:
     * </p>
     * 
     * <pre>
     * f(x) = max(0, x)
     * </pre>
     *
     * @param inOut
     *            input/output vector (read write)
     * @param size
     *            size of vector
     */
    public static native void ReLU(FloatBuffer inOut, int size);

    public static void ReLU(FloatBuffer inOut) {
        ReLU(inOut, inOut.limit());
    }

    /**
     * <p>
     * Exponential linear unit (ELU) function. Performs the operation in-place.
     * </p>
     * <p>
     * All input vector elements are transformed with function:
     * </p>
     * 
     * <pre>
     * f(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
     * </pre>
     *
     * @param inOut
     *            input/output vector (read write)
     * @param size
     *            size of vector
     * @param alpha
     *            varies the convergence value of the exponential function below zero
     */
    public static native void ELU(FloatBuffer inOut, int size, float alpha);

    public static void ELU(FloatBuffer inOut, float alpha) {
        ELU(inOut, inOut.limit(), alpha);
    }

    /**
     * <p>
     * Float matrix-vector multiplication. Destination memory is read and overwritten. Other buffers are read-only.
     * </p>
     * <p>
     * Operation:
     * </p>
     * 
     * <pre>
     * y = A * x + y
     * </pre>
     *
     * @param A
     *            input 2d matrix with logical dimensions: {@code ySize} x {@code xSize} (ro)
     * @param x
     *            input vector (ro)
     * @param y
     *            input/output vector (rw)
     * @param xSize
     *            size of input vector
     * @param ySize
     *            size of input/output vector
     * @see <a
     *      href=
     *      "http://www.netlib.org/lapack/explore-html/d6/d30/group__single__blas__level2.html#gafc92361b74c6d41c7e5afa0aa5d13ec9"
     *      >sgemv @ netlib lapack docs</a>
     */
    public static native void gemv(FloatBuffer A, FloatBuffer x, FloatBuffer y, int xSize, int ySize);

    /**
     * "Sizeless" version of {@link NeuralNetworkNativeOps#gemv}.
     */
    public static void gemv(FloatBuffer A, FloatBuffer x, FloatBuffer y) {
        gemv(A, x, y, x.limit(), y.limit());
    }

    /**
     * <p>
     * Float matrix-matrix multiplication. Destination memory is read and overwritten.
     * </p>
     * <p>
     * Operation:
     * </p>
     * 
     * <pre>
     * Y = A * B + Y
     * </pre>
     *
     * @param A
     *            input k x m matrix
     * @param B
     *            input m x n matrix
     * @param Y
     *            input/output k x n matrix
     * @see <a href=
     *      "http://www.netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3.html#gafe51bacb54592ff5de056acabd83c260"
     *      >sgemm @ netlib lapack docs</a>
     */
    public static native void gemm(FloatBuffer A, FloatBuffer B, FloatBuffer Y, int k, int m, int n);

    /**
     * "Sizeless" version of {@link NeuralNetworkNativeOps#gemm}.
     */
    public static void gemm(FloatBuffer A, FloatBuffer B, FloatBuffer Y) {
        gemm(A, B, Y,
                (int) Math.sqrt(A.limit() * Y.limit() / B.limit()),
                (int) Math.sqrt(A.limit() * B.limit() / Y.limit()),
                (int) Math.sqrt(B.limit() * Y.limit() / A.limit()));
    }

    /**
     * <p>
     * Forward operation for a single linear neural-network layer: Output contents is discarded and overwritten. Other
     * buffers are read-only.
     * </p>
     * 
     * <pre>
     * output = weights * input + biases
     * </pre>
     *
     * @param transpose
     *            whether {@code weights} should be transposed before multiplication
     * @param weights
     *            weights 2d matrix with logical dimensions: {@code outputSize} x {@code inputSize} (after optional
     *            transposition) (ro)
     * @param biases
     *            bias vector with size {@code outputSize} (ro)
     * @param input
     *            input vector with size {@code inputSize} (ro)
     * @param output
     *            output vector with size {@code outputSize} (write only)
     * @param inputSize
     *            size of input
     * @param outputSize
     *            size of output
     */
    public static native void linearForward(int transpose, FloatBuffer weights, FloatBuffer biases,
            FloatBuffer input, FloatBuffer output, int inputSize, int outputSize);

    /**
     * <p>
     * Forward operation for a single linear neural-network layer: Output contents is discarded and overwritten. Other
     * buffers are read-only.
     * </p>
     * 
     * <pre>
     * output = weights * input + biases
     * </pre>
     *
     * @param transpose
     *            whether {@code weights} should be transposed before multiplication
     * @param weights
     *            weights 2d matrix with logical dimensions: {@code outputSize} x {@code inputSize} (after optional
     *            transposition) (ro)
     * @param biases
     *            bias vector with size {@code outputSize} (ro)
     * @param input
     *            input vector with size {@code inputSize} (ro)
     * @param output
     *            output vector with size {@code outputSize} (write only)
     */
    public static void linearForward(int transpose, FloatBuffer weights, FloatBuffer biases, FloatBuffer input,
            FloatBuffer output) {
        linearForward(transpose, weights, biases, input, output, input.limit(), output.limit());
    }

    /**
     * <p>
     * Forward operation for a single linear neural-network layer: Output contents is discarded and overwritten. Other
     * buffers are read-only.
     * </p>
     * 
     * <pre>
     * output = input * weights + biases
     * </pre>
     *
     * @param transpose
     *            whether {@code weights} should be transposed before multiplication
     * @param weights
     *            weights 2d matrix with logical dimensions: {@code inputRowSize} x {@code outputRowSize} (after
     *            optional transposition)(ro)
     * @param biases
     *            bias vector with size {@code outputRowSize} (ro)
     * @param input
     *            input matrix with size {@code inputRowSize} x {@code batchSize} (ro). Values from one row should
     *            occupy consecutive memory
     *            cells
     * @param output
     *            output where matrix with size {@code outputSize} x {@code batchSize} will be written (write only)
     * @param inputRowSize
     *            number of floats in each row of the input
     * @param outputRowSize
     *            number of floats in each row of the output
     * @param batchSize
     *            number of rows in input, thus in output as well
     */
    public static native void linearBatchForward(int transpose, FloatBuffer weights, FloatBuffer biases,
            FloatBuffer input, FloatBuffer output, int inputRowSize, int outputRowSize, int batchSize);

    /**
     * Performs native side BLAS sgemv operation and checks if results are correct.
     * Used only for the module development purposes, doesn't make part of a regular usage pattern.
     */
    static native int test();
}
