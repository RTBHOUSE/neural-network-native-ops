#include <stdio.h>
#include <cblas.h>
#include <math.h>

#define TRANSPOSE 0
#define NO_TRANSPOSE 1

static const float ALPHA = 1.0, BETA = 1.0, ONE = 1.0;
static const int X_INC = 1, Y_INC = 1;

/**
 * Rectified linear unit (ReLU) function. Performs the operation in-place.
 *
 * All input vector elements are transformed with the function:
 *   f(x) = max(0, x)
 */
inline void ReLU(float *inOut, const int size) {
    int i;
    for(i=0; i<size; i++) {
        inOut[i] = inOut[i] < 0 ? 0 : inOut[i];
    }
}

/**
 * Exponential linear unit (ELU) function. Performs the operation in-place.
 *
 * All input vector elements are transformed with the function:
 *
 *   f(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
 */
inline void ELU(float *inOut, const int size, const float alpha) {
    int i;
    for(i=0; i < size; i++) {
        if (inOut[i] < 0) {
            inOut[i] = (expf(inOut[i]) - 1) * alpha;
        }
    }
}

/**
 * Float matrix-vector multiplication. Destination memory is read and overwritten.
 *
 *  A - input matrix
 *  x - input vector
 *  y - input/output vector
 *
 * Operation:
 *   y = A * x + y
 *
 * @see http://www.netlib.org/lapack/explore-html/d6/d30/group__single__blas__level2.html#gafc92361b74c6d41c7e5afa0aa5d13ec9
 */
inline void gemv(const float *A, const float *x, float *y, const int xSize, const int ySize) {
    cblas_sgemv( CblasRowMajor, CblasNoTrans, ySize, xSize, ALPHA, A, xSize, x, X_INC, BETA, y, Y_INC);
}


/**
 * Float matrix-matrix multiplication. Destination memory is read and overwritten.
 *
 *  A - input        m x k     matrix
 *  B - input            k x n matrix
 *  Y - input/output m   x   n matrix
 *
 * Operation:
 *   Y = A * B + Y
 *
 */
inline void gemm(const float *A, const float *B, float *Y, const int m, const int n, const int k) {
    cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, ONE, A, k, B, n, ONE, Y, n);
}

/**
 * Forward operation for a single linear neural-network layer
 *
 * Operation:
 *   output = weights(T) * input + biases
 *
 * (T) - optionally transposed
 */
inline void linearForward(int transposeWeights, const float *weights, const float *biases, const float *input, float *output,
        const int inputSize, const int outputSize) {

    memcpy(output, biases, outputSize * sizeof(float));
    if (transposeWeights == TRANSPOSE) {
        cblas_sgemv( CblasRowMajor, CblasTrans, inputSize, outputSize, ALPHA, weights, outputSize, input, X_INC, BETA, output, Y_INC);
    } else {
        cblas_sgemv( CblasRowMajor, CblasNoTrans, outputSize, inputSize, ALPHA, weights, inputSize, input, X_INC, BETA, output, Y_INC);
    }
}

/**
 * Forward operation for a single linear neural-network layer. Each input row should occupy consecutive memory cells.
 *
 * Operation:
 *   output = input * weights(T) + biases
 *
 * (T) - optionally transposed
 */
inline void linearBatchForward(int transposeWeights, const float *weights, const float *biases, const float *input, float *output,
        const int inputRowSize, const int outputRowSize, const int batchSize) {
    float *tmp = output;
    for(int i=0; i<batchSize; i++) {
        memcpy(tmp, biases, outputRowSize * sizeof(float));
        tmp += outputRowSize;
    }
    if (transposeWeights == TRANSPOSE) {
        cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasTrans, batchSize, outputRowSize, inputRowSize, ONE, input, inputRowSize, weights, inputRowSize, ONE, output, outputRowSize);
    } else {
        cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, batchSize, outputRowSize, inputRowSize, ONE, input, inputRowSize, weights, outputRowSize, ONE, output, outputRowSize);
    }
}

///////////////
// T E S T S //
///////////////
void pr(const char *name, const float *arr, const int size) {
    int i;
    printf(">>     %s = [", name);
    for(i=0; i<size; i++) {
        printf("%5.2f%s", arr[i], i<size-1 ? ", " : " ]");
    }
    printf("\n");
}

int test() {
    const int xS = 2, yS = 3;
    float A[] = { 1, 2,
                  3, 4,
                  2, 3};
    float x[] = {-1, 3};
    float y[] = {3, 2, 1};
    pr("A", A, xS * yS);
    pr("x", x, xS);
    pr("y before", y, yS);
    gemv(A, x, y, xS, yS);
    //cblas_sgemv ( CblasRowMajor, CblasNoTrans, yS, xS, 1, &A[1], 3, x, 1, 1, y, 1 );
    pr("y after", y, yS);
    if (y[0] != 8 || y[1] != 11 || y[2] != 8 ) {
        printf("gBLAS sgemv test failed: y should be [8,11,8]\n");
        return -1;
    }
    printf("gBLAS sgemv test OK\n");
    return 0;
}
