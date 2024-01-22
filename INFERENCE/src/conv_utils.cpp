#include <algorithm>
#include <cblas.h>
#include <vector>

extern "C" void im2col(const float* _image, float* _im2col, /*zeros*/
    const int b, const int ic, const int ih, const int iw,
    const int kh, const int kw,
    const int sh, const int sw,
    const int dh, const int dw,
    const int pad_top, const int pad_bottom, const int pad_left, const int pad_right)
{
    const int dkh = kh + (kh - 1) * (dh - 1);
    const int dkw = kw + (kw - 1) * (dw - 1);
    const int oh = (ih + pad_top + pad_bottom - dkh) / sh + 1;
    const int ow = (iw + pad_left + pad_right - dkw) / sw + 1;

    // image: [b, ic, ih, iw]
    // im2col: [b, oh, ow, ic, kh, kw]

    // 不考虑 pad 时
    // im2col[i0, i1, i2, i3, i4, i5] = image[i0, i3, i1 * sh + i4 * dh, i2 * sw + i5 * dw]
    const int image_step2 = iw;
    const int image_step1 = ih * image_step2;
    const int image_step0 = ic * image_step1;

    const int dimage_step2 = image_step2 * dh;

    const int im2col_step4 = kw;
    const int im2col_step3 = kh * im2col_step4;
    const int im2col_step2 = ic * im2col_step3;
    const int im2col_step1 = ow * im2col_step2;
    const int im2col_step0 = oh * im2col_step1;

    // i1 * sh + i4 * dh: [0, ih + pad_top + pad_bottom)
    // i2 * sw + i5 * dw: [0, iw + pad_left + pad_right)
    const float* pimage = _image;
    float* pim2col = _im2col;
    for (int i0 = 0; i0 < b; ++i0) {
        for (int i1 = 0; i1 < oh; ++i1) {
            const int i4min = std::max(0, (pad_top - i1 * sh + dh - 1) / dh);
            const int i4max = std::min(kh, (ih + pad_top - i1 * sh + dh - 1) / dh);
            const int ostep4_min = i4min * im2col_step4;
            const int isteph_min = (i1 * sh + i4min * dh - pad_top) * image_step2;
            for (int i2 = 0; i2 < ow; ++i2) {
                const int i5min = std::max(0, (pad_left - i2 * sw + dw - 1) / dw);
                const int i5max = std::min(kw, (iw + pad_left - i2 * sw + dw - 1) / dw);
                const int istepw_min = i2 * sw + i5min * dw - pad_left;
                int istepc = 0;
                for (int i3 = 0; i3 < ic; ++i3) {
                    int ostep4 = ostep4_min;
                    int isteph = isteph_min;
                    for (int i4 = i4min; i4 < i4max; ++i4) {
                        cblas_scopy(i5max - i5min, pimage + istepc + isteph + istepw_min, dw,
                            pim2col + ostep4 + i5min, 1);
                        ostep4 += im2col_step4;
                        isteph += dimage_step2;
                    }
                    istepc += image_step1;
                    pim2col += im2col_step3;
                }
            } // i2
        } // i1
        pimage += image_step0;
    } // i0
}

extern "C" void conv2d(const float* image, const float* weight, const float* bias,
    float* result, /*empty*/
    const int b, const int ic, const int ih, const int iw,
    const int oc, const int kh, const int kw,
    const int sh, const int sw, const int dh, const int dw,
    const int pad_top, const int pad_bottom, const int pad_left, const int pad_right)
{
    const int dkh = kh + (kh - 1) * (dh - 1);
    const int dkw = kw + (kw - 1) * (dw - 1);
    const int oh = (ih + pad_top + pad_bottom - dkh) / sh + 1;
    const int ow = (iw + pad_left + pad_right - dkw) / sw + 1;
    std::vector<float> v_im2col(b * oh * ow * ic * kh * kw);
    im2col(image, v_im2col.data(), b, ic, ih, iw, kh, kw, sh, sw, dh, dw,
        pad_top, pad_bottom, pad_left, pad_right);

    // im2col: [b, oh * ow, ic * ih * iw] -> [b, N, K]
    // weight: [oc, ic * ih * iw] -> [M, K]
    // result: [b, M, N]
    const int M = oc;
    const int N = oh * ow;
    const int K = ic * kh * kw;

    const int LDA = K;
    const int LDB = K;
    const int LDC = N;

    const int im2col_batch_step = N * K;
    const int result_batch_step = M * N;

    const float beta = bias ? 1.f : 0.f;
    const float* pim2col = v_im2col.data();
    float* presult = result;
    for (int i = 0; i < b; ++i) {
        if (bias) {
            for (int j = 0; j < oc; ++j)
                std::fill_n(presult + j * N, N, bias[j]);
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K,
            1.f, weight, LDA, pim2col, LDB, beta, presult, LDC);
        pim2col += im2col_batch_step;
        presult += result_batch_step;
    }
}
