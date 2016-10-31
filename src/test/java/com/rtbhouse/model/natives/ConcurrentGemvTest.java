package com.rtbhouse.model.natives;

import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.gemv;
import static com.rtbhouse.model.natives.NeuralNetworkNativeOpsTest.matrixFB;

import java.nio.FloatBuffer;

import org.junit.Test;

/**
 * This test checks that OpenBLAS is compiled with NUM_THREADS define large enough to handle numThreads
 * concurrent matrix-vector multiplications.
 */
public class ConcurrentGemvTest {
    private static int numThreads = 200;
    private static int numIterations = 5000;
    private static int dim1 = 500;
    private static int dim2 = 300;

    static {
        // This is needed to trigger `blas_memory_alloc()` that causes:
        // 'Program is Terminated. Because you tried to allocate too many memory regions' error.
        assert (dim1 + dim2) * Float.BYTES > 2048;
    }

    static class GemvThread extends Thread {
        FloatBuffer matrix = matrixFB(new float[dim1 * dim2]);
        FloatBuffer vector1 = matrixFB(new float[dim1]);
        FloatBuffer vector2 = matrixFB(new float[dim2]);

        @Override
        public void run() {
            for (int i = 0; i < numIterations; ++i) {
                gemv(matrix, vector1, vector2);
            }
        }
    }

    @Test
    public void testConcurrentGemv() throws InterruptedException {
        GemvThread[] threads = new GemvThread[numThreads];

        for (int i = 0; i < numThreads; ++i) {
            threads[i] = new GemvThread();
        }

        for (int i = 0; i < numThreads; ++i) {
            threads[i].start();
        }

        for (int i = 0; i < numThreads; ++i) {
            threads[i].join();
        }
    }
}
