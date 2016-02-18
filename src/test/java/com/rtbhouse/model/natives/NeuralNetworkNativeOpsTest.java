package com.rtbhouse.model.natives;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import org.junit.Assert;
import org.junit.Test;

import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.ReLU;
import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.gemv;
import static com.rtbhouse.model.natives.NeuralNetworkNativeOps.linearForward;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

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

    public static FloatBuffer allocateDirectFloatBufferOf(float... src) {
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
        linearForward(heapA, heapY, heapX, heapOutput);
        // then
        assertArrayEquals(expectedAbyXplusY, heapOutput.array(), MAX_ERROR);
    }

    @Test
    public void testDirectFloatBuffersLinearForward() {
        // when
        linearForward(directA, directY, directX, directOutput);
        // then
        assertArrayEquals(expectedAbyXplusY, getArrayFrom(directOutput), MAX_ERROR);
    }
}
