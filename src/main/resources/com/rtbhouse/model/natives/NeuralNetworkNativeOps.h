#include <stdio.h>
#include <cblas.h>
#include <math.h>

enum NNNOTranspose {
    TRANSPOSE = 0,
    NO_TRANSPOSE = 1
};

static const float ALPHA = 1.0;
static const float BETA = 1.0;
static const float ONE = 1.0;
static const int X_INC = 1;
static const int Y_INC = 1;

/**
 * In-place applies the rectified linear unit (ReLU) function to the first `endExclusive` input vector elements:
 *
 *   ReLU(x) = max(0, x)
 */
inline void ReLU(float *inOut, const int endExclusive) {
    int i;
    for(i=0; i < endExclusive; i++) {
        if (inOut[i] < 0) {
            inOut[i] = 0;
        }
    }
}

/**
 * In-place applies the exponential linear unit (ELU) function to the first `endExclusive` input vector elements:
 *
 *   ELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
 */
inline void ELU(float *inOut, const int endExclusive, const float alpha) {
    int i;
    for(i=0; i < endExclusive; i++) {
        if (inOut[i] < 0) {
            inOut[i] = (expf(inOut[i]) - 1) * alpha;
        }
    }
}

/**
 * Applies a float matrix-vector multiplication with accumulation (gemv).
 *
 *   y = A * x + y
 *
 * Destination memory is read and overwritten. Other buffers are read-only.
 *
 *  A - input matrix        (n x m)
 *  x - input vector        (    m)
 *  y - input/output vector (n    )
 *
 * @see http://www.netlib.org/lapack/explore-html/d6/d30/group__single__blas__level2_gafc92361b74c6d41c7e5afa0aa5d13ec9
 */
inline void gemv(const float *A, const float *x, float *y, const int m, const int n) {
    cblas_sgemv( CblasRowMajor, CblasNoTrans, n, m, ALPHA, A, m, x, X_INC, BETA, y, Y_INC);
}


/**
 * Applies a float matrix-matrix multiplication with accumulation (gemm):
 *
 *   Y = A * B + Y
 *
 *  A - input matrix        (m x k    )
 *  B - input matrix        (    k x n)
 *  Y - input/output matrix (m   x   n)
 *
 * @see http://www.netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260
 */
inline void gemm(const float *A, const float *B, float *Y, const int m, const int n, const int k) {
    cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, ONE, A, k, B, n, ONE, Y, n);
}

/**
 * Forward operation for a single linear neural-network layer:
 *
 *   output = weights(T) * input + biases
 *
 * (T) - optionally transposed
 */
inline void linearForward(const NNNOTranspose transposeWeights, const float *weights, const float *biases,
        const float *input, float *output, const int inputSize, const int outputSize) {

    memcpy(output, biases, outputSize * sizeof(float));
    if (transposeWeights == TRANSPOSE) {
        cblas_sgemv( CblasRowMajor, CblasTrans,
            inputSize, outputSize, ALPHA, weights, outputSize, input, X_INC, BETA, output, Y_INC);
    } else {
        cblas_sgemv( CblasRowMajor, CblasNoTrans,
            outputSize, inputSize, ALPHA, weights, inputSize, input, X_INC, BETA, output, Y_INC);
    }
}

/**
 * Forward operation for a single linear neural-network layer. Each input row must occupy consecutive memory cells:
 *
 *   output = input * weights(T) + biases
 *
 * (T) - optionally transposed
 */
inline void linearBatchForward(const NNNOTranspose transposeWeights, const float *weights, const float *biases,
        const float *input, float *output, const int inputRowSize, const int outputRowSize, const int batchSize) {
    float *tmp = output;
    for(int i=0; i < batchSize; i++) {
        memcpy(tmp, biases, outputRowSize * sizeof(float));
        tmp += outputRowSize;
    }
    if (transposeWeights == TRANSPOSE) {
        cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasTrans, batchSize, outputRowSize, inputRowSize, ONE,
            input, inputRowSize, weights, inputRowSize, ONE, output, outputRowSize);
    } else {
        cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, batchSize, outputRowSize, inputRowSize, ONE,
            input, inputRowSize, weights, outputRowSize, ONE, output, outputRowSize);
    }
}
