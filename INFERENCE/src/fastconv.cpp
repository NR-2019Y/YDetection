#include <algorithm>
#ifdef MKL
#include <mkl/mkl.h>
#else
#include <cblas.h>
#endif
#include <vector>

// https://github.com/BVLC/caffe/blob/master/LICENSE

extern "C" void im2col_cpu(const float* _image, float* _im2col, /*fill*/
    const int bic, const int ih, const int iw,
    const int kh, const int kw,
    const int sh, const int sw,
    const int dh, const int dw,
    const int pad_top, const int pad_bottom, const int pad_left, const int pad_right,
    const int num_threads)
{
    const int dkh = kh + (kh - 1) * (dh - 1);
    const int dkw = kw + (kw - 1) * (dw - 1);
    const int oh = (ih + pad_top + pad_bottom - dkh) / sh + 1;
    const int ow = (iw + pad_left + pad_right - dkw) / sw + 1;

    // image: [bic, ih, iw]
    // im2col: [bic, kh, kw, oh, ow]

    // 不考虑pad时
    // foreach bic
    // im2col[i0, i1, i2, i3] = image[i0*dh+i2*sh, i1*dw+i3*sw]
    // const float* pimage = _image;
    // float* pim2col = _im2col;

    const int image_steph = iw;
    const int image_stepbic = ih * image_steph;
    const int dimage_steph = dh * image_steph;
    const int simage_steph = sh * image_steph;

    const int im2col_stepoh = ow;
    const int im2col_stepkw = oh * im2col_stepoh;
    const int im2col_stepkh = kw * im2col_stepkw;
    const int im2col_stepbic = kh * im2col_stepkh;

    const int pad_shift = -pad_top * image_steph - pad_left;
#pragma omp parallel for num_threads(num_threads)
    for (int q = 0; q < bic; ++q) {
        const float* pimage = _image + q * image_stepbic;
        float* pim2col = _im2col + q * im2col_stepbic;
        float* curr_pim2col = pim2col;
        for (int i0 = 0; i0 < kh; ++i0) {
            const int i2min = std::max(0, (pad_top - i0 * dh + sh - 1) / sh);
            const int i2max = std::min(oh, (ih + pad_top - i0 * dh + sh - 1) / sh);
            const int ostepoh_min = i2min * im2col_stepoh;
            const int istep_i2min = i2min * simage_steph;
            const int istep_i0 = i0 * dimage_steph;
            for (int i1 = 0; i1 < kw; ++i1) {
                const int i3min = std::max(0, (pad_left - i1 * dw + sw - 1) / sw);
                const int i3max = std::min(ow, (iw + pad_left - i1 * dw + sw - 1) / sw);
                int ostepoh_i3s = ostepoh_min + i3min;
                int istep_i0i1i2_i3s = istep_i0 + i1 * dw + istep_i2min + i3min * sw + pad_shift;
                for (int i2 = i2min; i2 < i2max; ++i2) {
                    cblas_scopy(i3max - i3min, pimage + istep_i0i1i2_i3s, sw, curr_pim2col + ostepoh_i3s, 1);
                    // cblas_scopy(i3max - i3min, pimage + (i0 * dh + i2 * sh - pad_top) * image_steph + i1 * dw + i3min * sw - pad_left, sw,
                    //     curr_pim2col + i2 * im2col_stepoh + i3min, 1);
                    istep_i0i1i2_i3s += simage_steph;
                    ostepoh_i3s += im2col_stepoh;
                }
                curr_pim2col += im2col_stepkw;
            }
        }
    }
}

extern "C" void conv2d_cpu(const float* _image, const float* _weight, const float* _bias, float* _result, /*empty*/
    const int b, const int ic, const int ih, const int iw,
    const int oc, const int kh, const int kw,
    const int sh, const int sw, const int dh, const int dw,
    const int pad_top, const int pad_bottom, const int pad_left, const int pad_right, const int num_threads)
{
    const int dkh = kh + (kh - 1) * (dh - 1);
    const int dkw = kw + (kw - 1) * (dw - 1);
    const int oh = (ih + pad_top + pad_bottom - dkh) / sh + 1;
    const int ow = (iw + pad_left + pad_right - dkw) / sw + 1;
    // image: [b, ic, ih, iw]
    // im2col: [b, ic, kh, kw, oh, ow]
    std::vector<float> v_im2col(b * ic * kh * kw * oh * ow, 0.f);
    im2col_cpu(_image, v_im2col.data(), b * ic, ih, iw, kh, kw, sh, sw, dh, dw,
        pad_top, pad_bottom, pad_left, pad_right, num_threads);

    const int im2col_stepb = ic * kh * kw * oh * ow;
    const int result_stepb = oc * oh * ow;

    const float* pim2col = v_im2col.data();
    float* presult = _result;

    // each batch:
    // weight: [oc, ic * kh * kw]
    // im2col: [ic * kh * kw, oh * ow]
    // -> [oc, oh * ow]
    const int GEMM_M = oc;
    const int GEMM_N = oh * ow;
    const int GEMM_K = ic * kh * kw;
    const int LDA = GEMM_K;
    const int LDB = GEMM_N;
    const int LDC = GEMM_N;

    const int cstep = oh * ow;

    const float beta = _bias ? 1.0f : 0.0f;
    for (int _ = 0; _ < b; ++_) {
        if (_bias) {
            for (int i = 0; i < oc; ++i)
                std::fill_n(presult + i * cstep, cstep, _bias[i]);
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, GEMM_M, GEMM_N, GEMM_K,
            1.f, _weight, LDA, pim2col, LDB, beta, presult, LDC);
        pim2col += im2col_stepb;
        presult += result_stepb;
    }
}
