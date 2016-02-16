package com.rtbhouse.model.natives;

import java.nio.FloatBuffer;

import org.bytedeco.javacpp.annotation.Platform;

import com.github.fommil.jni.JniLoader;

@Platform(include = "NeuralNetworkNativeOps.h", compiler = "fastfpu")
public final class NeuralNetworkNativeOps {

    static {
        JniLoader.load("lib/jniNeuralNetworkNativeOps.so");
    }

    private NeuralNetworkNativeOps() {
    }

    @SuppressWarnings({ "PMD.MethodNamingConventions", "checkstyle:MethodName" })
    public static native void ReLU(FloatBuffer inOut, int size);

    /**
     * Float matrix-by-vector multiplication. Destination memory is read and overwritten.
     *
     * Operation:
     * 
     * <pre>
     * y = A * x + y
     * </pre>
     *
     * @param A
     *            input matrix (read only) with dimensions: {@code xsize} x {@code ySize}
     * @param x
     *            input vector (ro)
     * @param y
     *            input/output vector (rw)
     * @param xSize
     *            size of input vector (ro)
     * @param ySize
     *            size of input/output vector (ro)
     *
     * @see <a
     *      href=
     *      "http://www.netlib.org/lapack/explore-html/d6/d30/group__single__blas__level2.html#gafc92361b74c6d41c7e5afa0aa5d13ec9"
     *      >sgemv @ netlib lapack docs</a>
     */
    @SuppressWarnings({ "PMD.MethodNamingConventions", "checkstyle:ParameterName" })
    public static native void gemv(FloatBuffer A, FloatBuffer x, FloatBuffer y, int xSize, int ySize);

    /**
     * Forward operation for single linear neural network layer:
     * 
     * <pre>
     * output = weights * input + biases
     * </pre>
     * 
     * Output contents are discarded and overwritten. Other buffers are read-only.
     */
    public static native void linearForward(FloatBuffer weights, FloatBuffer biases,
            FloatBuffer input, FloatBuffer output, int inputSize, int outputSize);

    /**
     * performs BLAS sgemv operation and checks if results are correct
     */
    public static native void test();

    public static void main(String[] args) throws Exception {
        test();
    }
}
