#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <vector>

extern "C" void im2col_cfirst(const float* _image, float* _im2col, /*-inf*/
    const int bic, const int ih, const int iw,
    const int kh, const int kw,
    const int sh, const int sw,
    const int dh, const int dw,
    const int pad_top, const int pad_bottom, const int pad_left, const int pad_right)
{
    const int dkh = kh + (kh - 1) * (dh - 1);
    const int dkw = kw + (kw - 1) * (dw - 1);
    const int oh = (ih + pad_top + pad_bottom - dkh) / sh + 1;
    const int ow = (iw + pad_left + pad_right - dkw) / sw + 1;

    // image: [bic, ih, iw]
    // im2col: [bic, oh, ow, kh, kw]
    const int image_steph = iw;
    const int image_stepbic = ih * image_steph;

    const int dimage_steph = dh * image_steph;

    const int im2col_stepkh = kw;
    const int im2col_stepow = kh * im2col_stepkh;
    const int im2col_stepoh = ow * im2col_stepow;
    // const int im2col_stepbic = oh * im2col_stepoh;

    const float* pimage = _image;
    float* pim2col = _im2col;
    for (int _ = 0; _ < bic; ++_) {
        // im2col[i0, i1, i2, i3] = image[i0 * sh + i2 * dh, i1 * sw + i3 * dw]
        for (int i0 = 0; i0 < oh; ++i0) {
            const int i2min = std::max(0, (pad_top - i0 * sh + dh - 1) / dh);
            const int i2max = std::min(kh, (ih + pad_top - i0 * sh + dh - 1) / dh);
            const int isteph_min = (i0 * sh + i2min * dh - pad_top) * image_steph;
            const int ostepkh_min = i2min * im2col_stepkh;
            for (int i1 = 0; i1 < ow; ++i1) {
                const int i3min = std::max(0, (pad_left - i1 * sw + dw - 1) / dw);
                const int i3max = std::min(kw, (iw + pad_left - i1 * sw + dw - 1) / dw);
                const int istepw_min = i1 * sw + i3min * dw - pad_left;
                int isteph = isteph_min;
                int ostepkh = ostepkh_min;
                for (int i2 = i2min; i2 < i2max; ++i2) {
                    cblas_scopy(i3max - i3min, pimage + isteph + istepw_min, dw,
                        pim2col + ostepkh + i3min, 1);
                    isteph += dimage_steph;
                    ostepkh += im2col_stepkh;
                }
                pim2col += im2col_stepow;
            }
        }
        pimage += image_stepbic;
        // pim2col += im2col_stepbic;
    }
}

extern "C" void maxpool2d(const float* image, float* result, /*empty*/
    const int bic, const int ih, const int iw,
    const int kh, const int kw,
    const int sh, const int sw,
    const int dh, const int dw,
    const int pad_top, const int pad_bottom, const int pad_left, const int pad_right)
{
    const int dkh = kh + (kh - 1) * (dh - 1);
    const int dkw = kw + (kw - 1) * (dw - 1);
    const int oh = (ih + pad_top + pad_bottom - dkh) / sh + 1;
    const int ow = (iw + pad_left + pad_right - dkw) / sw + 1;

    // image: [bic, ih, iw]
    // im2col: [bic, oh, ow, kh, kw]
    std::vector<float> v_im2col(bic * oh * ow * kh * ow, -INFINITY);

    im2col_cfirst(image, v_im2col.data(), bic, ih, iw, kh, kw, sh, sw, dh, dw,
        pad_top, pad_bottom, pad_left, pad_right);

    const int total = bic * oh * ow;
    const int im2col_step = kh * kw;

    const float* pim2col = v_im2col.data();
    float* presult = result;
    for (int i = 0; i < total; ++i) {
        *presult++ = *std::max_element(pim2col, pim2col + im2col_step);
        pim2col += im2col_step;
    }
}
