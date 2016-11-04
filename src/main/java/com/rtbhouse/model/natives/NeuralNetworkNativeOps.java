package com.rtbhouse.model.natives;

import java.nio.FloatBuffer;

import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.MemberGetter;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.ValueGetter;

import com.github.fommil.jni.JniLoader;

/**
 * <p>
 * Utility class with few operations useful when dealing with neural networks. Operations are implemented at native side
 * with BLAS behind the scenes.
 * </p>
 * Supports only single precision floating point numbers. Both heap and direct float buffers are supported but
 * an order of magnitude performance boost is achieved when using direct buffers.
 *
 * @author Piotr Chromiec
 */
@Platform(include = "NeuralNetworkNativeOps.h", compiler = "fastfpu")
public final class NeuralNetworkNativeOps {

    static {
        JniLoader.load("com/rtbhouse/model/natives/libjniNeuralNetworkNativeOps.so");
    }

    private static native @MemberGetter @Const int TRANSPOSE();

    private static native @ValueGetter @Const int NO_TRANSPOSE();

    public enum Trans {
        TRANSPOSE(TRANSPOSE()), NO_TRANSPOSE(NO_TRANSPOSE());

        private int value;

        Trans(int value) {
            this.value = value;
        }

        private int value() {
            return value;
        }
    }

    private NeuralNetworkNativeOps() {
    }

    /**
     * In-place applies the rectified linear unit (ReLU) function to the first {@code endExclusive} input elements:
     *
     * <pre>
     * ReLU(x) = max(0, x)
     * </pre>
     *
     * @param inOut
     *            input/output vector (read write)
     * @param endExclusive
     *            index immediately past the last index to process
     */
    public static void ReLU(FloatBuffer inOut, int endExclusive) {
        if (endExclusive > inOut.limit() || endExclusive < 0) {
            throw new IndexOutOfBoundsException();
        }

        nativeReLU(inOut, endExclusive);
    }

    /**
     * Applies the {@link NeuralNetworkNativeOps#ReLU} for whole input vector in-place.
     *
     * @param inOut
     *            input/output vector (read write)
     */
    public static void ReLU(FloatBuffer inOut) {
        nativeReLU(inOut, inOut.limit());
    }

    private static native @Name("ReLU") void nativeReLU(FloatBuffer inOut, int endExclusive);

    /**
     * In-place applies the exponential linear unit (ELU) function to the first {@code endExclusive} input vector
     * elements:
     *
     * <pre>
     * ELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
     * </pre>
     *
     * @param inOut
     *            input/output vector (read write)
     * @param endExclusive
     *            index immediately past the last index to process
     * @param alpha
     *            varies the convergence value of the exponential function below zero
     */
    public static void ELU(FloatBuffer inOut, int endExclusive, float alpha) {
        if (endExclusive > inOut.limit() || endExclusive < 0) {
            throw new IndexOutOfBoundsException();
        }

        nativeELU(inOut, endExclusive, alpha);
    }

    /**
     * Applies the {@link NeuralNetworkNativeOps#ELU} for whole input vector in-place.
     *
     * @param inOut
     *            input/output vector (read write)
     * @param alpha
     *            varies the convergence value of the exponential function below zero
     */
    public static void ELU(FloatBuffer inOut, float alpha) {
        nativeELU(inOut, inOut.limit(), alpha);
    }

    private static native @Name("ELU") void nativeELU(FloatBuffer inOut, int endExclusive, float alpha);

    /**
     * Applies a float matrix-vector multiplication with accumulation (gemv):
     *
     * <pre>
     * y = A * x + y
     * </pre>
     *
     * Destination memory is read and overwritten. Other buffers are read-only.
     *
     * @param A
     *            input matrix with logical dimensions: {@code n} x {@code m} (ro)
     * @param x
     *            input vector (ro)
     * @param y
     *            input/output vector (rw)
     * @param m
     *            index immediately past the last index to process in {@code x} and number of logical columns in
     *            {@code A}
     * @param n
     *            index immediately past the last index to process in {@code y} and number of logical rows in {@code A}
     * @see <a
     *      href=
     *      "http://www.netlib.org/lapack/explore-html/d6/d30/group__single__blas__level2_gafc92361b74c6d41c7e5afa0aa5d13ec9"
     *      >sgemv @ netlib lapack docs</a>
     */
    public static void gemv(FloatBuffer A, FloatBuffer x, FloatBuffer y, int m, int n) {
        if (m > x.limit() || n > y.limit() || m * n > A.limit() || m < 0 || n < 0) {
            throw new IndexOutOfBoundsException();
        }

        nativeGemv(A, x, y, m, n);
    }

    /**
     * Applies the {@link NeuralNetworkNativeOps#gemv} to the whole incoming data in-place.
     * 
     * @param A
     *            input matrix with logical dimensions: {@code y.limit()} x {@code x.limit()} (ro)
     * @param x
     *            input vector (ro)
     * @param y
     *            input/output vector (rw)
     */
    public static void gemv(FloatBuffer A, FloatBuffer x, FloatBuffer y) {
        if (x.limit() * y.limit() != A.limit()) {
            throw new IllegalArgumentException();
        }

        nativeGemv(A, x, y, x.limit(), y.limit());
    }

    private static native @Name("gemv") void nativeGemv(FloatBuffer A, FloatBuffer x, FloatBuffer y, int xSize,
            int ySize);

    /**
     * Applies a float matrix-matrix multiplication with accumulation (gemm):
     * 
     * <pre>
     * Y = A * B + Y
     * </pre>
     *
     * Destination memory is read and overwritten. Other buffers are read-only.
     * 
     * @param A
     *            input matrix with logical dimensions: {@code m} x {@code k} (ro)
     * @param B
     *            input matrix with logical dimensions: {@code k} x {@code n} (ro)
     * @param Y
     *            input/output matrix with logical dimensions: {@code m} x {@code n} (rw)
     * @param m
     *            number of logical rows in {@code A} and {@code Y}
     * @param n
     *            number of logical columns in {@code B} and {@code Y}
     * @param k
     *            number of logical columns in {@code A} and logical rows in {@code B}
     * @see <a href=
     *      "http://www.netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3.html#gafe51bacb54592ff5de056acabd83c260"
     *      >sgemm @ netlib lapack docs</a>
     */
    public static void gemm(FloatBuffer A, FloatBuffer B, FloatBuffer Y, int m, int n, int k) {
        if (m * k > A.limit() || k * n > B.limit() || m * n > Y.limit() || k < 0 || m < 0 || n < 0) {
            throw new IndexOutOfBoundsException();
        }

        nativeGemm(A, B, Y, m, n, k);
    }

    private static native @Name("gemm") void nativeGemm(FloatBuffer A, FloatBuffer B, FloatBuffer Y, int m, int n, int k);

    /**
     * Applies a linear transformation to the incoming data:
     *
     * <pre>
     * output = weights * input + biases
     * </pre>
     * 
     * Output contents are discarded and overwritten. Other buffers are read-only.
     * 
     * @param transposeWeights
     *            whether {@code weights} should be transposed before multiplication
     * @param weights
     *            weights matrix with logical dimensions: {@code inputSize} x {@code outputSize} if
     *            {@code transposeWeights == TRANSPOSE}, reversed otherwise (ro)
     * @param biases
     *            biases vector (ro)
     * @param input
     *            input vector (ro)
     * @param output
     *            output vector (write only)
     * @param inputSize
     *            index immediately past the last index to process in {@code input} and number of logical rows in
     *            {@code weights} if {@code transposeWeights == TRANSPOSE}, columns otherwise.
     * @param outputSize
     *            index immediately past the last index to process in {@code output} and {@code biases}; also number of
     *            logical columns in {@code weights} if {@code transposeWeights == TRANSPOSE}, rows otherwise.
     */
    public static void linearForward(Trans transposeWeights, FloatBuffer weights, FloatBuffer biases,
            FloatBuffer input, FloatBuffer output, int inputSize, int outputSize) {

        if (inputSize > input.limit() || outputSize > output.limit() || outputSize > biases.limit()
                || outputSize * inputSize > weights.limit() || inputSize < 0 || outputSize < 0) {
            throw new IndexOutOfBoundsException();
        }

        nativeLinearForward(transposeWeights.value(), weights, biases, input, output, inputSize, outputSize);
    }

    /**
     * Applies the {@link NeuralNetworkNativeOps#linearForward} to the whole incoming data in-place.
     * 
     * @param transposeWeights
     *            whether {@code weights} should be transposed before multiplication
     * @param weights
     *            weights matrix with logical dimensions: {@code input.limit()} x {@code output.limit()} if
     *            {@code transposeWeights == TRANSPOSE}, reversed otherwise (ro)
     * @param biases
     *            bias vector with size {@code output.limit()} (ro)
     * @param input
     *            input vector (ro)
     * @param output
     *            output vector (write only)
     */
    public static void linearForward(Trans transposeWeights, FloatBuffer weights, FloatBuffer biases,
            FloatBuffer input, FloatBuffer output) {

        if (input.limit() * output.limit() != weights.limit() || output.limit() != biases.limit()) {
            throw new IllegalArgumentException();
        }

        nativeLinearForward(transposeWeights.value(), weights, biases, input, output, input.limit(), output.limit());
    }

    private static native @Name("linearForward") void nativeLinearForward(
            @Cast("NNNOTranspose") int transposeWeights, FloatBuffer weights, FloatBuffer biases, FloatBuffer input,
            FloatBuffer output, int inputSize, int outputSize);

    /**
     * Applies a linear transformation to the incoming data:
     *
     * <pre>
     * output = weights * input + biases
     * </pre>
     *
     * Output contents are discarded and overwritten. Other buffers are read-only.
     *
     * @param transposeWeights
     *            whether {@code weights} should be transposed before multiplication
     * @param weights
     *            weights matrix with logical dimensions: {@code outputRowSize} x {@code inputRowSize} if
     *            {@code transposeWeights == TRANSPOSE}, reversed otherwise (ro)
     * @param biases
     *            bias vector with size {@code outputRowSize} (ro)
     * @param input
     *            input matrix with size {@code batchSize} x {@code inputRowSize}; values from one row should
     *            occupy consecutive memory cells (ro)
     * @param output
     *            output matrix with size {@code batchSize} x {@code outputRowSize} (write only)
     * @param inputRowSize
     *            number of logical columns in {@code input} and logical columns in {@code weights} if
     *            {@code transposeWeights == TRANSPOSE}, rows otherwise
     * @param outputRowSize
     *            number of logical columns in {@code output} and logical rows in {@code weights} if
     *            {@code transposeWeights == TRANSPOSE}, columns otherwise
     * @param batchSize
     *            number of logical rows in {@code input} and {@code output} to process
     */
    public static void linearBatchForward(Trans transposeWeights, FloatBuffer weights, FloatBuffer biases,
            FloatBuffer input, FloatBuffer output, int inputRowSize, int outputRowSize, int batchSize) {

        if (inputRowSize * batchSize > input.limit() || outputRowSize * batchSize > output.limit()
                || outputRowSize > biases.limit() || inputRowSize * outputRowSize > weights.limit()
                || outputRowSize < 0 || inputRowSize < 0 || batchSize < 0) {
            throw new IndexOutOfBoundsException();
        }

        nativeLinearBatchForward(transposeWeights.value(),
                weights, biases, input, output, inputRowSize, outputRowSize, batchSize);
    }

    private static native @Name("linearBatchForward") void nativeLinearBatchForward(
            @Cast("NNNOTranspose") int transposeWeights, FloatBuffer weights, FloatBuffer biases, FloatBuffer input,
            FloatBuffer output, int inputRowSize, int outputRowSize, int batchSize);
}
