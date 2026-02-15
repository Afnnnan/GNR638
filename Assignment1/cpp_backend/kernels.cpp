#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" {

void relu_forward(const float* x, float* y, int size) {
    for (int i = 0; i < size; ++i) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

void relu_backward(const float* x, const float* grad_out, float* grad_x, int size) {
    for (int i = 0; i < size; ++i) {
        grad_x[i] = x[i] > 0.0f ? grad_out[i] : 0.0f;
    }
}

void linear_forward(
    const float* x,
    const float* w,
    const float* b,
    float* y,
    int batch,
    int in_features,
    int out_features
) {
    for (int n = 0; n < batch; ++n) {
        for (int o = 0; o < out_features; ++o) {
            float sum = b ? b[o] : 0.0f;
            for (int i = 0; i < in_features; ++i) {
                sum += x[n * in_features + i] * w[o * in_features + i];
            }
            y[n * out_features + o] = sum;
        }
    }
}

void linear_backward_input(
    const float* grad_out,
    const float* w,
    float* grad_x,
    int batch,
    int in_features,
    int out_features
) {
    std::fill(grad_x, grad_x + batch * in_features, 0.0f);
    for (int n = 0; n < batch; ++n) {
        for (int i = 0; i < in_features; ++i) {
            float sum = 0.0f;
            for (int o = 0; o < out_features; ++o) {
                sum += grad_out[n * out_features + o] * w[o * in_features + i];
            }
            grad_x[n * in_features + i] = sum;
        }
    }
}

void linear_backward_weight(
    const float* x,
    const float* grad_out,
    float* grad_w,
    int batch,
    int in_features,
    int out_features
) {
    std::fill(grad_w, grad_w + out_features * in_features, 0.0f);
    for (int o = 0; o < out_features; ++o) {
        for (int i = 0; i < in_features; ++i) {
            float sum = 0.0f;
            for (int n = 0; n < batch; ++n) {
                sum += grad_out[n * out_features + o] * x[n * in_features + i];
            }
            grad_w[o * in_features + i] = sum;
        }
    }
}

void linear_backward_bias(const float* grad_out, float* grad_b, int batch, int out_features) {
    std::fill(grad_b, grad_b + out_features, 0.0f);
    for (int o = 0; o < out_features; ++o) {
        float sum = 0.0f;
        for (int n = 0; n < batch; ++n) {
            sum += grad_out[n * out_features + o];
        }
        grad_b[o] = sum;
    }
}

void maxpool2d_forward(
    const float* x,
    float* y,
    int32_t* indices,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int kernel,
    int stride
) {
    const int out_h = (in_h - kernel) / stride + 1;
    const int out_w = (in_w - kernel) / stride + 1;
    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float best = -1e30f;
                    int32_t best_idx = -1;
                    for (int kh = 0; kh < kernel; ++kh) {
                        for (int kw = 0; kw < kernel; ++kw) {
                            const int ih = oh * stride + kh;
                            const int iw = ow * stride + kw;
                            const int idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                            const float val = x[idx];
                            if (val > best) {
                                best = val;
                                best_idx = idx;
                            }
                        }
                    }
                    const int out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;
                    y[out_idx] = best;
                    indices[out_idx] = best_idx;
                }
            }
        }
    }
}

void maxpool2d_backward(
    const float* grad_out,
    const int32_t* indices,
    float* grad_x,
    int grad_out_size,
    int grad_x_size
) {
    std::fill(grad_x, grad_x + grad_x_size, 0.0f);
    for (int i = 0; i < grad_out_size; ++i) {
        const int32_t src = indices[i];
        if (src >= 0 && src < grad_x_size) {
            grad_x[src] += grad_out[i];
        }
    }
}

void conv2d_forward(
    const float* x,
    const float* w,
    const float* b,
    float* y,
    int batch,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding
) {
    const int out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    const int out_w = (in_w + 2 * padding - kernel_w) / stride + 1;

    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = b ? b[oc] : 0.0f;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                const int ih = oh * stride + kh - padding;
                                const int iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    const int x_idx = ((n * in_channels + ic) * in_h + ih) * in_w + iw;
                                    const int w_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                                    sum += x[x_idx] * w[w_idx];
                                }
                            }
                        }
                    }
                    const int y_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
                    y[y_idx] = sum;
                }
            }
        }
    }
}

void conv2d_backward_input(
    const float* grad_out,
    const float* w,
    float* grad_x,
    int batch,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding
) {
    const int out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    const int out_w = (in_w + 2 * padding - kernel_w) / stride + 1;
    std::fill(grad_x, grad_x + batch * in_channels * in_h * in_w, 0.0f);

    for (int n = 0; n < batch; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    const float go = grad_out[((n * out_channels + oc) * out_h + oh) * out_w + ow];
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                const int ih = oh * stride + kh - padding;
                                const int iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    const int gx_idx = ((n * in_channels + ic) * in_h + ih) * in_w + iw;
                                    const int w_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                                    grad_x[gx_idx] += go * w[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void conv2d_backward_weight(
    const float* x,
    const float* grad_out,
    float* grad_w,
    int batch,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride,
    int padding
) {
    const int out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    const int out_w = (in_w + 2 * padding - kernel_w) / stride + 1;
    std::fill(grad_w, grad_w + out_channels * in_channels * kernel_h * kernel_w, 0.0f);

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    float sum = 0.0f;
                    for (int n = 0; n < batch; ++n) {
                        for (int oh = 0; oh < out_h; ++oh) {
                            for (int ow = 0; ow < out_w; ++ow) {
                                const int ih = oh * stride + kh - padding;
                                const int iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    const int x_idx = ((n * in_channels + ic) * in_h + ih) * in_w + iw;
                                    const int go_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
                                    sum += x[x_idx] * grad_out[go_idx];
                                }
                            }
                        }
                    }
                    const int gw_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                    grad_w[gw_idx] = sum;
                }
            }
        }
    }
}

void conv2d_backward_bias(const float* grad_out, float* grad_b, int batch, int out_channels, int out_h, int out_w) {
    std::fill(grad_b, grad_b + out_channels, 0.0f);
    for (int oc = 0; oc < out_channels; ++oc) {
        float sum = 0.0f;
        for (int n = 0; n < batch; ++n) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    const int idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
                    sum += grad_out[idx];
                }
            }
        }
        grad_b[oc] = sum;
    }
}

}
