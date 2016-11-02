package com.rtbhouse.model.natives;

import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.ELU;
import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.ReLU;
import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.Trans.NO_TRANSPOSE;
import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.Trans.TRANSPOSE;
import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.gemm;
import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.gemv;
import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.linearBatchForward;
import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.linearForward;
import static org.junit.Assert.assertArrayEquals;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import org.junit.Test;

public class NeuralNetworkNativeOpsTest {
    static final float MAX_ERROR = 1e-6f;

    private final FloatBuffer heapA = matrixFB(1f / 3, 2, 3, 4, 2, 3);
    private final FloatBuffer heapX = matrixFB(-1, 3);
    private final FloatBuffer heapY = matrixFB(3, 2, 1f / 3);
    private final FloatBuffer heapOutput = matrixFB(-1, -1, -1);

    private final FloatBuffer directA = allocateDirectFloatBufferOf(1f / 3, 2, 3, 4, 2, 3);
    private final FloatBuffer directX = allocateDirectFloatBufferOf(-1, 3);
    private final FloatBuffer directY = allocateDirectFloatBufferOf(3, 2, 1f / 3);
    private final FloatBuffer directOutput = allocateDirectFloatBufferOf(-1, -1, -1);

    private final float[] expectedReLUx = matrix(0, 3);
    private final float[] expectedReLUFirstTwoOutput = matrix(0, 0, -1);
    private final float[] expectedELUx2 = matrix(2 / (float) Math.E - 2, 3);
    private final float[] expectedELUx2FirstOutput = matrix(2 / (float) Math.E - 2, -1, -1);
    private final float[] expectedAbyXplusY = matrix(8f + (2f / 3), 11, 7f + (1f / 3));
    private final float[] expectedTransposedAbyXplusY = matrix(14f + (2f / 3), 6, 6f + (1f / 3));
    private final float[] expectedAbyFirstValXplusY = matrix(2f + (2f / 3), 0, -3 + (1f / 3));

    private static FloatBuffer allocateDirectFloatBufferOf(float... src) {
        return ByteBuffer
                .allocateDirect(src.length * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(src);
    }

    private static float[] getArrayFrom(FloatBuffer buffer) {
        float[] array = new float[buffer.limit()];
        buffer.rewind();
        buffer.get(array);
        return array;
    }

    @Test
    public void shouldReLUwithHeapFloatBuffers() {
        // when
        ReLU(heapOutput, 2);
        // then
        assertArrayEquals(expectedReLUFirstTwoOutput, heapOutput.array(), MAX_ERROR);
    }

    @Test
    public void shouldReLUwithHeapFloatBuffersSizeless() {
        // when
        ReLU(heapX);
        // then
        assertArrayEquals(expectedReLUx, heapX.array(), MAX_ERROR);
    }

    @Test
    public void shouldReLUwithDirectFloatBuffers() {
        // when
        ReLU(directX);
        // then
        assertArrayEquals(expectedReLUx, getArrayFrom(directX), MAX_ERROR);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void shouldReLUUnderflowThrow() {
        ReLU(heapX, -1);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void shouldReLUOverflowThrow() {
        ReLU(heapX, heapX.limit() + 1);
    }

    @Test
    public void shouldELUwithHeapFloatBuffers() {
        // when
        ELU(heapOutput, 1, 2);
        // then
        assertArrayEquals(expectedELUx2FirstOutput, heapOutput.array(), MAX_ERROR);
    }

    @Test
    public void shouldELUwithHeapFloatBuffersSizeless() {
        // when
        ELU(heapX, 2);
        // then
        assertArrayEquals(expectedELUx2, heapX.array(), MAX_ERROR);
    }

    @Test
    public void shouldELUwithDirectFloatBuffers() {
        // when
        ELU(directX, 2);
        // then
        assertArrayEquals(expectedELUx2, getArrayFrom(directX), MAX_ERROR);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void shouldELUUnderflowThrow() {
        ELU(heapX, -1, 0);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void shouldELUOverflowThrow() {
        ELU(heapX, heapX.limit() + 1, 0);
    }

    @Test
    public void shouldMultiplyMatrixByVectorWithHeapFloatBuffers() {
        // when
        gemv(heapA, heapX, heapY, 1, heapY.limit());
        // then
        assertArrayEquals(expectedAbyFirstValXplusY, heapY.array(), MAX_ERROR);
    }

    @Test
    public void shouldMultiplyMatrixByVectorWithHeapFloatBuffersSizeless() {
        // when
        gemv(heapA, heapX, heapY);
        // then
        assertArrayEquals(expectedAbyXplusY, heapY.array(), MAX_ERROR);
    }

    @Test
    public void shouldMultiplyMatrixByVectorWithDirectFloatBuffers() {
        // when
        gemv(directA, directX, directY);
        // then
        assertArrayEquals(expectedAbyXplusY, getArrayFrom(directY), MAX_ERROR);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void shouldGemvUnderflowThrow() {
        gemv(directA, directX, directY, -1, 0);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void shouldGemvOverflowThrow() {
        gemv(heapA, heapX, heapY, 1, heapY.limit() + 1);
    }

    @Test
    public void shouldHeapFloatBuffersLinearForward() {
        // when
        linearForward(NO_TRANSPOSE, heapA, heapY, heapX, heapOutput, 1, heapY.limit());
        // then
        assertArrayEquals(expectedAbyFirstValXplusY, heapOutput.array(), MAX_ERROR);
    }

    @Test
    public void shouldHeapFloatBuffersLinearForwardSizeless() {
        // when
        linearForward(NO_TRANSPOSE, heapA, heapY, heapX, heapOutput);
        // then
        assertArrayEquals(expectedAbyXplusY, heapOutput.array(), MAX_ERROR);
    }

    @Test
    public void shouldHeapFloatBuffersLinearForwardSizelessTransposed() {
        // when
        linearForward(TRANSPOSE, heapA, heapY, heapX, heapOutput);
        // then
        assertArrayEquals(expectedTransposedAbyXplusY, heapOutput.array(), MAX_ERROR);
    }

    @Test
    public void shouldDirectFloatBuffersLinearForward() {
        // when
        linearForward(NO_TRANSPOSE, directA, directY, directX, directOutput);
        // then
        assertArrayEquals(expectedAbyXplusY, getArrayFrom(directOutput), MAX_ERROR);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void shouldLinearForwardUnderflowThrow() {
        linearForward(NO_TRANSPOSE, directA, directY, directX, directOutput, -1, 0);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void shouldLinearForwardOverflowThrow() {
        linearForward(NO_TRANSPOSE, heapA, heapY, heapX, heapOutput, heapX.limit() + 1, 0);
    }

    @Test
    public void shouldMultiplyTwoMatrices() {
        // given
        FloatBuffer a = matrixFB(
                1f / 3, 1, -1,
                2, 4, -1); // ignored row
        FloatBuffer b = matrixFB(
                1f / 5, 1,
                7, 49,
                -2, -3);
        FloatBuffer c = matrixFB(
                0, 0);

        // when
        gemm(a, b, c, 1, 2, 3);

        // then
        assertArrayEquals(
                matrix(9.06666666f, 52.33333333f),
                getArrayFrom(c),
                MAX_ERROR);
    }

    @Test
    public void shouldMultiplyTwoMatricesSizeless() {
        // given
        FloatBuffer a = matrixFB(
                1f / 3, 1, -1,
                2, 4, -1);
        FloatBuffer b = matrixFB(
                1f / 5, 1,
                7, 49,
                -2, -3);
        FloatBuffer c = matrixFB(
                0, 0,
                0, 0);

        // when
        gemm(a, b, c, 2, 2, 3);

        // then
        assertArrayEquals(
                matrix(9.06666666f, 52.33333333f,
                        30.4000f, 201.00000f),
                getArrayFrom(c),
                MAX_ERROR);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void shouldMultiplyTwoMatricesUnderflowThrow() {
        gemm(heapA, heapA, heapX, -1, 0, 0);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void shouldMultiplyTwoMatricesOverflowThrow() {
        gemm(heapA, heapA, heapX, heapA.limit() + 1, 1, 0);
    }

    @Test
    public void shouldForwardLinearOnBatch() {
        // given
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

        // when
        linearBatchForward(NO_TRANSPOSE, weights, biases, input, output, 2, 3, 5);

        // then
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
        // given
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

        // when
        linearBatchForward(TRANSPOSE, weights, biases, input, output, 2, 3, 5);

        // then
        assertArrayEquals(
                matrix(31f / 15 + 0.1f, 4.4f, 9.3f,
                        301f / 3 + 0.1f, 203.2f, 461.7f,
                        31f / 15 + 0.1f, 4.4f, 9.3f,
                        31f / 15 + 0.1f, 4.4f, 9.3f,
                        301f / 3 + 0.1f, 203.2f, 461.7f),
                getArrayFrom(output),
                MAX_ERROR);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void shouldForwardLinearOnBatchUnderflowThrow() {
        linearBatchForward(NO_TRANSPOSE, heapA, heapY, heapX, heapY, -1, 0, 0);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void shouldForwardLinearOnBatchOverflowThrow() {
        linearBatchForward(NO_TRANSPOSE, heapA, heapY, heapX, heapY, heapA.limit() + 1, 1, 0);
    }

    private float[] matrix(float... values) {
        return values;
    }

    static FloatBuffer matrixFB(float... values) {
        return FloatBuffer.wrap(values);
    }
}
