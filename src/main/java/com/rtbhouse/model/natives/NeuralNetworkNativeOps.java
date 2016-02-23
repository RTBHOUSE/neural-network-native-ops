package com.rtbhouse.model.natives;

import java.nio.FloatBuffer;

import org.bytedeco.javacpp.annotation.Platform;

import com.github.fommil.jni.JniLoader;

/**
 * Util class with few operations useful when dealing with neural networks: ReLU, linearForward and simple
 * matrix-by-vector multiplication. Operations are implemented on native side with BLAS behind the scenes.
 *
 * Supports only single precision floating point numbers. Both heap and direct float buffers are supported but
 * order of magnitude performance boost is achieved when using direct buffers.
 *
 * @author Piotr Chromiec
 */
@Platform(include = "NeuralNetworkNativeOps.h", compiler = "fastfpu")
public final class NeuralNetworkNativeOps {

    static {
        JniLoader.load("com/rtbhouse/model/natives/libjniNeuralNetworkNativeOps.so");
    }

    private NeuralNetworkNativeOps() {
    }

    /**
     * Rectified linear unit (ReLU) function. Do its operation in-place.
     *
     * All input vector elements are transformed with function:
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
        ReLU(inOut, inOut.capacity());
    }

    /**
     * Float matrix-by-vector multiplication. Destination memory is read and overwritten.
     * Contents of {@code y} are read and overwritten. Other buffers are read-only.
     *
     * Operation:
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
     *
     * @see <a
     *      href=
     *      "http://www.netlib.org/lapack/explore-html/d6/d30/group__single__blas__level2.html#gafc92361b74c6d41c7e5afa0aa5d13ec9"
     *      >sgemv @ netlib lapack docs</a>
     */
    public static native void gemv(FloatBuffer A, FloatBuffer x, FloatBuffer y, int xSize, int ySize);

    public static void gemv(FloatBuffer A, FloatBuffer x, FloatBuffer y) {
        gemv(A, x, y, x.capacity(), y.capacity());
    }

    /**
     * Forward operation for single linear neural network layer:
     * Output contents are discarded and overwritten. Other buffers are read-only.
     * 
     * <pre>
     * output = weights * input + biases
     * </pre>
     * 
     * @param weights
     *            weights 2d matrix with logical dimensions: {@code outputSize} x {@code inputSize} (ro)
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
    public static native void linearForward(FloatBuffer weights, FloatBuffer biases,
            FloatBuffer input, FloatBuffer output, int inputSize, int outputSize);

    public static void linearForward(FloatBuffer weights, FloatBuffer biases, FloatBuffer input, FloatBuffer output) {
        linearForward(weights, biases, input, output, input.capacity(), output.capacity());
    }

    /**
     * Performs native side BLAS sgemv operation and checks if results are correct.
     */
    static native int test();
}
