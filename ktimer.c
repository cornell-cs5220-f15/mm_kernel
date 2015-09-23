#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define MIN_RUNS 4
#define MIN_SECS 0.25

// External DGEMM kernel
void kdgemm(const double* A, const double* B, double* C);

// Routines to convert between column major and internal matrix formats
void to_kdgemm_A(int ldA, const double* A, double* Ak);
void to_kdgemm_B(int ldB, const double* B, double* Bk);
void from_kdgemm_C(int ldC, const double* Ck, double* C);

// Matrix sizes for external kernel
extern int DIM_M;
extern int DIM_N;
extern int DIM_K;

void matrix_init(double *A, int M, int N)
{
    for (int i = 0; i < M*N; ++i) 
        A[i] = drand48();
}

void matrix_clear(double *C, int M, int N)
{
    memset(C, 0, M * N * sizeof(double));
}


/*
 * Time the matrix multiply
 */
double time_dgemm(const double *A, const double *B, double *C)
{
    double secs = -1.0;
    double mflops_sec;
    int num_iterations = MIN_RUNS;
    while (secs < MIN_SECS) {
        matrix_clear(C, DIM_M, DIM_N);
        double start = omp_get_wtime();
        for (int i = 0; i < num_iterations; ++i) {
            kdgemm(A, B, C);
        }
        double finish = omp_get_wtime();
        double mflops = 2.0 * num_iterations * DIM_M * DIM_N * DIM_K / 1.0e6;
        secs = finish-start;
        mflops_sec = mflops / secs;
        num_iterations *= 2;
    }
    return mflops_sec;
}


/*
 * Print the computed output (C), the reference output (A*B),
 * and the difference C-A*B.  This is useful for debugging issues
 * with memory layout conversions.
 */
double diff_kdgemm(double* A, double* B, double* C)
{
    printf("Computed:\n");
    for (int i = 0; i < DIM_M; ++i) {
        for (int j = 0; j < DIM_N; ++j)
            printf(" %g", C[i+j*DIM_M]);
        printf("\n");
    }
    printf("\nReference:\n");
    for (int i = 0; i < DIM_M; ++i) {
        for (int j = 0; j < DIM_N; ++j) {
            double cij = 0;
            for (int k = 0; k < DIM_K; ++k)
                cij += A[i+k*DIM_M]*B[k+j*DIM_K];
            printf(" %g", cij);
        }
        printf("\n");
    }
    printf("\nDiff:\n");
    for (int i = 0; i < DIM_M; ++i) {
        for (int j = 0; j < DIM_N; ++j) {
            double cij = 0;
            for (int k = 0; k < DIM_K; ++k)
                cij += A[i+k*DIM_M]*B[k+j*DIM_K];
            printf(" % 0.0e", cij-C[i+j*DIM_M]);
        }
        printf("\n");
    }
}


/*
 * Return the max absolute element of C-A*B.  This should be on
 * the order of a few times machine epsilon (roughly 1e-16)
 * if the multiply is functioning correctly.
 */
double check_kdgemm(double* A, double* B, double* C)
{
    double max_diff = 0;
    for (int i = 0; i < DIM_M; ++i) {
        for (int j = 0; j < DIM_N; ++j) {
            double cij = 0;
            for (int k = 0; k < DIM_K; ++k)
                cij += A[i+k*DIM_M]*B[k+j*DIM_K];
            double diff = fabs(cij-C[i+j*DIM_M]);
            if (diff > max_diff)
                max_diff = diff;
        }
    }
    return max_diff;
}


/*
 * Run a basic test and timing trial.
 */
int main(int argc, char** argv)
{
    // Allocate space for ordinary column-major matrices
    double* A = malloc(DIM_M * DIM_K * sizeof(double));
    double* B = malloc(DIM_K * DIM_N * sizeof(double));
    double* C = malloc(DIM_M * DIM_N * sizeof(double));

    // Allocate aligned scratch space for use by the kernel
    double* Ak = _mm_malloc(DIM_M * DIM_K * sizeof(double), 16);
    double* Bk = _mm_malloc(DIM_K * DIM_N * sizeof(double), 16);
    double* Ck = _mm_malloc(DIM_M * DIM_N * sizeof(double), 16);

    // Initialize the input matrices and convert to kernel format
    matrix_init(A, DIM_M, DIM_K);
    matrix_init(B, DIM_K, DIM_N);
    to_kdgemm_A(DIM_M, A, Ak);
    to_kdgemm_B(DIM_K, B, Bk);

    // Clear the kernel scratch output, run the kernel, convert to col major
    matrix_clear(Ck, DIM_M, DIM_N);
    kdgemm(Ak, Bk, Ck);
    from_kdgemm_C(DIM_M, Ck, C);

    // Check for agreement
    double max_diff = check_kdgemm(A, B, C);

    // Print kernel dimensions, megaflop rate, and error from check
    printf("%u,%u,%u,%lg,%0.0e\n", DIM_M, DIM_K, DIM_N, 
           time_dgemm(Ak, Bk, Ck),
           max_diff);

    // Free kernel matrix space
    _mm_free(Ck);
    _mm_free(Bk);
    _mm_free(Ak);

    // Free argument matrix space
    free(C);
    free(B);
    free(A);

    return 0;
}

