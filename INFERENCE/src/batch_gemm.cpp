#include <cblas.h>

extern "C" void batch_sgemm(const int batch_size,
    const int batch_step_a, const int batch_step_b, const int batch_step_c,
    const int trans_a, const int trans_b, 
    const int M, const int N, const int K,
    const float alpha, const float* A, const int LDA, 
    const float* B, const int LDB,
    const float beta, float* C, const int LDC)
{
    for (int i = 0; i < batch_size; ++i) {
        cblas_sgemm(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
            M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC);
        A += batch_step_a;
        B += batch_step_b;
        C += batch_step_c;
    }
}
