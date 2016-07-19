package com.rtbhouse.model.natives;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import org.bytedeco.javacpp.annotation.Platform;

import com.github.fommil.jni.JniLoader;

/**
 * Utility class that delegates Field-aware Factorization Machines prediction to native code.
 */
@Platform(include = "FactorizationMachinesNativeOps.h", compiler = "fastfpu")
public final class FactorizationMachinesNativeOps {

    /** FFM factor size. */
    public final static int FACTOR_SIZE = 4;

    static {
        JniLoader.load("com/rtbhouse/model/natives/libjniFactorizationMachinesNativeOps.so");
    }

    private FactorizationMachinesNativeOps() {
    }

    /**
     * Performs Field-aware Factorization Machines prediction.
     *
     * Supports only single precision floating point numbers. Both heap and direct float buffers are supported but
     * an order of magnitude performance boost is achieved when using direct buffers.
     *
     * @param weights
     *            Factorization Machines model weights
     * @param features
     *            Input features
     *
     * @return predicted value
     */
    public static float ffmPredict(FloatBuffer weights, IntBuffer features) {
        return ffmPredict(weights, features.capacity(), features);
    }

    private static native float ffmPredict(FloatBuffer weights, int numFields, IntBuffer features);
}
