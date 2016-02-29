#include <stdio.h>
#include <cblas.h>

static const float ALPHA = 1.0, BETA = 1.0;
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
 * Forward operation for a single linear neural-network layer
 *
 * Operation:
 *   output = weights * input + biases
 */
inline void linearForward(const float *weights, const float *biases, const float *input, float *output,
        const int inputSize, const int outputSize) {

    memcpy(output, biases, outputSize * sizeof(float));
    gemv(weights, input, output, inputSize, outputSize);
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
