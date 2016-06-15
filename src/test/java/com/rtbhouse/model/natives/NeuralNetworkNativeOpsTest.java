package com.rtbhouse.model.natives;

import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.NO_TRANSPOSE;
import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.ReLU;
import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.TRANSPOSE;
import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.gemv;
import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.linearForward;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import org.junit.Test;

public class NeuralNetworkNativeOpsTest {
    static final float MAX_ERROR = 1e-6f;

    private final FloatBuffer heapA = FloatBuffer.wrap(new float[] { 1f / 3, 2, 3, 4, 2, 3 });
    private final FloatBuffer heapX = FloatBuffer.wrap(new float[] { -1, 3 });
    private final FloatBuffer heapY = FloatBuffer.wrap(new float[] { 3, 2, 1f / 3 });
    private final FloatBuffer heapOutput = FloatBuffer.wrap(new float[] { -1, -1, -1 });

    private final FloatBuffer directA = allocateDirectFloatBufferOf(1f / 3, 2, 3, 4, 2, 3);
    private final FloatBuffer directX = allocateDirectFloatBufferOf(-1, 3);
    private final FloatBuffer directY = allocateDirectFloatBufferOf(3, 2, 1f / 3);
    private final FloatBuffer directOutput = allocateDirectFloatBufferOf(-1, -1, -1);

    private final float[] expectedReLUx = new float[] { 0, 3 };
    private float[] expectedAbyXplusY = new float[] { 8f + (2f / 3), 11, 7f + (1f / 3) };

    private static FloatBuffer allocateDirectFloatBufferOf(float... src) {
        return ByteBuffer
                .allocateDirect(src.length * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(src);
    }

    private static float[] getArrayFrom(FloatBuffer buffer) {
        float[] array = new float[buffer.capacity()];
        buffer.rewind();
        buffer.get(array);
        return array;
    }

    @Test
    public void nativeTestInvoke() {
        assertEquals(0, NeuralNetworkNativeOps.test());
    }

    @Test
    public void testHeapFloatBuffersReLU() {
        // when
        ReLU(heapX);
        // then
        assertArrayEquals(expectedReLUx, heapX.array(), MAX_ERROR);
    }

    @Test
    public void testDirectFloatBuffersReLU() {
        // when
        ReLU(directX);
        // then
        assertArrayEquals(expectedReLUx, getArrayFrom(directX), MAX_ERROR);
    }

    @Test
    public void testHeapFloatBuffersGemv() {
        // when
        gemv(heapA, heapX, heapY);
        // then
        assertArrayEquals(expectedAbyXplusY, heapY.array(), MAX_ERROR);
    }

    @Test
    public void testDirectFloatBuffersGemv() {
        // when
        gemv(directA, directX, directY);
        // then
        assertArrayEquals(expectedAbyXplusY, getArrayFrom(directY), MAX_ERROR);
    }

    @Test
    public void testHeapFloatBuffersLinearForward() {
        // when
        linearForward(NO_TRANSPOSE, heapA, heapY, heapX, heapOutput);
        // then
        assertArrayEquals(expectedAbyXplusY, heapOutput.array(), MAX_ERROR);
    }

    @Test
    public void testDirectFloatBuffersLinearForward() {
        // when
        linearForward(NO_TRANSPOSE, directA, directY, directX, directOutput);
        // then
        assertArrayEquals(expectedAbyXplusY, getArrayFrom(directOutput), MAX_ERROR);
    }

    @Test
    public void shouldMultiplyTwoMatrices() {
        //given
        FloatBuffer a = matrixFB(
                1f / 3, 1, -1,
                2, 4, -1);
        FloatBuffer b = matrixFB(
                1f / 5, 1,
                7, 49,
                -2, -3);
        FloatBuffer c = FloatBuffer.wrap(new float[] { 0, 0,
                0, 0 });

        //when
        NeuralNetworkNativeOps.gemm(a, b, c, 2, 2, 3);

        //then
        assertArrayEquals(
                matrix(9.06666666f, 52.33333333f,
                        30.4000f, 201.00000f),
                getArrayFrom(c),
                MAX_ERROR);
    }

    @Test
    public void shouldForwardLinearOnBatch() {
        //given
        FloatBuffer weights = matrixFB(
                1f / 3, 1, 3,
                2, 4, 9);
        FloatBuffer biases = matrixFB(0.1f, 0.2f, -0.3f);

        FloatBuffer input = matrixFB(
                1f / 5, 1,
                7, 49,
                1f / 5, 1,
                1f / 5, 1,
                7, 49);
        FloatBuffer output = matrixFB(
                1, 2, 3,
                3, 4, 5,
                5, 6, 6,
                7, 8, 7,
                9, 10, -1);

        //when
        NeuralNetworkNativeOps.linearBatchForward(NO_TRANSPOSE, weights, biases, input, output, 2, 3, 5);

        //then
        assertArrayEquals(
                matrix(31f / 15 + 0.1f, 4.4f, 9.3f,
                        301f / 3 + 0.1f, 203.2f, 461.7f,
                        31f / 15 + 0.1f, 4.4f, 9.3f,
                        31f / 15 + 0.1f, 4.4f, 9.3f,
                        301f / 3 + 0.1f, 203.2f, 461.7f),
                getArrayFrom(output),
                MAX_ERROR);
    }

    @Test
    public void shouldForwardLinearOnBatchWithTranspose() {
        //given
        FloatBuffer weights = matrixFB(
                1f / 3, 2,
                1, 4,
                3, 9);
        FloatBuffer biases = matrixFB(0.1f, 0.2f, -0.3f);

        FloatBuffer input = matrixFB(
                1f / 5, 1,
                7, 49,
                1f / 5, 1,
                1f / 5, 1,
                7, 49);
        FloatBuffer output = matrixFB(
                1, 2, 3,
                3, 4, 5,
                5, 6, 6,
                7, 8, 7,
                9, 10, -1);

        //when
        NeuralNetworkNativeOps.linearBatchForward(TRANSPOSE, weights, biases, input, output, 2, 3, 5);

        //then
        assertArrayEquals(
                matrix(31f / 15 + 0.1f, 4.4f, 9.3f,
                        301f / 3 + 0.1f, 203.2f, 461.7f,
                        31f / 15 + 0.1f, 4.4f, 9.3f,
                        31f / 15 + 0.1f, 4.4f, 9.3f,
                        301f / 3 + 0.1f, 203.2f, 461.7f),
                getArrayFrom(output),
                MAX_ERROR);
    }

    @Test
    public void shouldForwardLinear() {
        //given
        FloatBuffer weights = matrixFB(
                1f / 3, 2,
                1, 4,
                3, 9);
        FloatBuffer biases = matrixFB(0.1f, 0.2f, -0.3f); //vector
        FloatBuffer input = matrixFB(1f / 5, 1); //vector
        FloatBuffer output = matrixFB(1, 2, 3); //vector

        //when
        NeuralNetworkNativeOps.linearForward(NO_TRANSPOSE, weights, biases, input, output);

        //then
        assertArrayEquals(
                matrix(31f / 15 + 0.1f, 4.4f, 9.3f),
                getArrayFrom(output),
                MAX_ERROR);
    }

    @Test
    public void shouldForwardLinearWithTranspose() {
        //given
        FloatBuffer weights = matrixFB(
                1f / 3, 1, 3,
                2, 4, 9);
        FloatBuffer biases = matrixFB(0.1f, 0.2f, -0.3f); //vector
        FloatBuffer input = matrixFB(1f / 5, 1); //vector
        FloatBuffer output = matrixFB(1, 2, 3); //vector

        //when
        NeuralNetworkNativeOps.linearForward(TRANSPOSE, weights, biases, input, output);

        //then
        assertArrayEquals(
                matrix(31f / 15 + 0.1f, 4.4f, 9.3f),
                getArrayFrom(output),
                MAX_ERROR);
    }

    private float[] matrix(float... values) {
        return values;
    }

    private FloatBuffer matrixFB(float... values) {
        return FloatBuffer.wrap(values);
    }
}
