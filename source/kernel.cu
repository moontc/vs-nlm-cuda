// based on KNLMeansCL by Khanattila

#include "common.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <type_traits>

template <typename T>
__host__ __device__
static inline T square(T x) {
    return x * x;
}

template <typename T, int results_per_thread = 1>
__global__
__launch_bounds__(128)
static void distance_horizontal(
    float * __restrict__ buffer,
    const T * __restrict__ src,
    const T * __restrict__ neighbor_src,
    int width, int height, int image_stride, int buffer_stride,
    int offset_x, int offset_y,
    int block_radius, ChannelMode channels,
    float inv_divisor
) {

    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= height) {
        return ;
    }

    auto scale = [inv_divisor](float x) -> float {
        if constexpr (!std::is_floating_point_v<T>) {
            x *= square(inv_divisor);
        }
        return x;
    };

    extern __shared__ float local_buffer[]; // shape: (block_h, results_per_thread * block_w + 2 * nlm_s)
    // ignores bank conflicts here for higher occupancy
    // at most (32 / blockDim.x) smem requests generated per instruction
    int local_buffer_stride = results_per_thread * blockDim.x + 2 * block_radius;

    auto clamp_x = [width](int coord) { return min(max(coord, 0), width - 1); };
    auto clamp_y = [height](int coord) { return min(max(coord, 0), height - 1); };

    for (int i = threadIdx.x; i < results_per_thread * blockDim.x + 2 * block_radius; i += blockDim.x) {
        int x = -block_radius + blockIdx.x * results_per_thread * blockDim.x;

        switch (channels) {
            case ChannelMode::Y: {
                auto center = static_cast<float>(src[y * image_stride + clamp_x(x + i)]);
                auto neighbor = static_cast<float>(neighbor_src[clamp_y(y + offset_y) * image_stride + clamp_x(clamp_x(x + i) + offset_x)]);
                auto diff = center - neighbor;
                local_buffer[threadIdx.y * local_buffer_stride + i] = scale(square(diff) * 3);
                break;
            }
            case ChannelMode::UV: {
                auto center_u = static_cast<float>(src[y * image_stride + clamp_x(x + i)]);
                auto neighbor_u = static_cast<float>(neighbor_src[(clamp_y(y + offset_y)) * image_stride + clamp_x(clamp_x(x + i) + offset_x)]);
                auto diff_u = center_u - neighbor_u;
                auto center_v = static_cast<float>(src[(height + y) * image_stride + clamp_x(x + i)]);
                auto neighbor_v = static_cast<float>(neighbor_src[(height + clamp_y(y + offset_y)) * image_stride + clamp_x(clamp_x(x + i) + offset_x)]);
                auto diff_v = center_v - neighbor_v;
                local_buffer[threadIdx.y * local_buffer_stride + i] = scale((square(diff_u) + square(diff_v)) * 1.5f);
                break;
            }
            case ChannelMode::YUV: {
                auto center_y = static_cast<float>(src[y * image_stride + clamp_x(x + i)]);
                auto neighbor_y = static_cast<float>(neighbor_src[(clamp_y(y + offset_y)) * image_stride + clamp_x(clamp_x(x + i) + offset_x)]);
                auto diff_y = center_y - neighbor_y;
                auto center_u = static_cast<float>(src[(height + y) * image_stride + clamp_x(x + i)]);
                auto neighbor_u = static_cast<float>(neighbor_src[(height + clamp_y(y + offset_y)) * image_stride + clamp_x(clamp_x(x + i) + offset_x)]);
                auto diff_u = center_u - neighbor_u;
                auto center_v = static_cast<float>(src[(2 * height + y) * image_stride + clamp_x(x + i)]);
                auto neighbor_v = static_cast<float>(neighbor_src[(2 * height + clamp_y(y + offset_y)) * image_stride + clamp_x(clamp_x(x + i) + offset_x)]);
                auto diff_v = center_v - neighbor_v;
                local_buffer[threadIdx.y * local_buffer_stride + i] = scale(square(diff_y) + square(diff_u) + square(diff_v));
                break;
            }
            case ChannelMode::RGB: {
                auto center_r = static_cast<float>(src[y * image_stride + clamp_x(x + i)]);
                auto neighbor_r = static_cast<float>(neighbor_src[(clamp_y(y + offset_y)) * image_stride + clamp_x(clamp_x(x + i) + offset_x)]);
                auto diff_r = center_r - neighbor_r;
                auto center_g = static_cast<float>(src[(height + y) * image_stride + clamp_x(x + i)]);
                auto neighbor_g = static_cast<float>(neighbor_src[(height + clamp_y(y + offset_y)) * image_stride + clamp_x(clamp_x(x + i) + offset_x)]);
                auto diff_g = center_g - neighbor_g;
                auto center_b = static_cast<float>(src[(2 * height + y) * image_stride + clamp_x(x + i)]);
                auto neighbor_b = static_cast<float>(neighbor_src[(2 * height + clamp_y(y + offset_y)) * image_stride + clamp_x(clamp_x(x + i) + offset_x)]);
                auto diff_b = center_b - neighbor_b;
                auto weight = (center_r + neighbor_r) / 6 * inv_divisor;
                local_buffer[threadIdx.y * local_buffer_stride + i] = scale(
                    (2.0f / 3.0f + weight) * square(diff_r) +
                    (4.0f / 3.0f) * square(diff_g) +
                    (1.0f - weight) * square(diff_b)
                );
                break;
            }
            default:
                __builtin_unreachable();
        }
    }

    __syncwarp();

    int x = blockIdx.x * results_per_thread * blockDim.x + threadIdx.x;
    int local_x = threadIdx.x + block_radius;
    for (int loop = 0; loop < results_per_thread; loop++, x += blockDim.x, local_x += blockDim.x) {
        if (x >= width) {
            return ;
        }

        float sum = 0.0f;
        for (int i = -block_radius; i <= block_radius; i++) {
            sum += local_buffer[threadIdx.y * local_buffer_stride + (local_x + i)];
        }

        buffer[y * buffer_stride + x] = sum;
    }
}

template <int results_per_thread = 1>
static void distance_horizontal_dispatch(
    float * buffer,
    const void * src, int src_offset, int neighbor_src_offset,
    int width, int height, int image_stride, int buffer_stride,
    int offset_x, int offset_y,
    int block_radius, ChannelMode channels,
    dim3 grid, dim3 block, size_t dyn_smem_size, cudaStream_t stream,
    bool is_float, int bits_per_sample
) {
    if (is_float) {
        if (bits_per_sample == 32) {
            distance_horizontal<float, results_per_thread><<<grid, block, dyn_smem_size, stream>>>(
                buffer,
                static_cast<const float *>(src) + src_offset,
                static_cast<const float *>(src) + neighbor_src_offset,
                width, height, image_stride, buffer_stride,
                offset_x, offset_y,
                block_radius, channels, 0.0f
            );
        }
    } else {
        float inv_divisor = 1.0f / ((1 << bits_per_sample) - 1);

        if (bits_per_sample <= 8) {
            distance_horizontal<uint8_t, results_per_thread><<<grid, block, dyn_smem_size, stream>>>(
                buffer,
                static_cast<const uint8_t *>(src) + src_offset,
                static_cast<const uint8_t *>(src) + neighbor_src_offset,
                width, height, image_stride, buffer_stride,
                offset_x, offset_y,
                block_radius, channels, inv_divisor
            );
        } else if (bits_per_sample <= 16) {
            distance_horizontal<uint16_t, results_per_thread><<<grid, block, dyn_smem_size, stream>>>(
                buffer,
                static_cast<const uint16_t *>(src) + src_offset,
                static_cast<const uint16_t *>(src) + neighbor_src_offset,
                width, height, image_stride, buffer_stride,
                offset_x, offset_y,
                block_radius, channels, inv_divisor
            );
        }
    }
}

template <int results_per_thread = 1>
__global__
__launch_bounds__(128)
static void vertical(
    float * __restrict__ buffer_out,
    const float * __restrict__ buffer,
    int width, int height, int buffer_stride,
    int block_radius,
    float h2_inv_norm,
    int wmode
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width) {
        return ;
    }

    extern __shared__ float local_buffer[]; // shape: (results_per_thread * block_h + 2 * nlm_s, block_w)

    auto clamp_y = [height](int coord) { return min(max(coord, 0), height - 1); };

    for (int i = threadIdx.y; i < results_per_thread * blockDim.y + 2 * block_radius; i += blockDim.y) {
        int y = -block_radius + blockIdx.y * results_per_thread * blockDim.y;
        local_buffer[i * blockDim.x + threadIdx.x] = buffer[clamp_y(y + i) * buffer_stride + x];
    }

    __syncthreads();

    int y = blockIdx.y * results_per_thread * blockDim.y + threadIdx.y;
    int local_y = threadIdx.y + block_radius;
    for (int loop = 0; loop < results_per_thread; loop++, y += blockDim.y, local_y += blockDim.y) {
        if (y >= height) {
            return ;
        }

        float sum = 0.0f;
        for (int i = -block_radius; i <= block_radius; i++) {
            sum += local_buffer[(local_y + i) * blockDim.x + threadIdx.x];
        }

        float val;
        if (wmode == 0) {
            val = expf(-sum * h2_inv_norm);
        } else {
            val = fdimf(1.0f, sum * h2_inv_norm);
            if (wmode >= 2) {
                val *= val;
            }
            if (wmode == 3) {
                val *= val;
                val *= val;
            }
        }

        buffer_out[y * buffer_stride + x] = val;
    }
}

template <typename T, int num_planes>
__global__
__launch_bounds__(128)
static void accumulation_grouped(
    float * __restrict__ wdst,
    float * __restrict__ weight,
    float * __restrict__ max_weight,
    const T * __restrict__ src,
    const float * __restrict__ buffer_bwd_base,
    const float * __restrict__ buffer_fwd_base,
    const int4 * __restrict__ offsets,
    int group_start, int group_count,
    int frame_buffer_elems,
    int width, int height, int image_stride, int buffer_stride
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return ;
    }

    auto clamp_x = [width](int coord) { return min(max(coord, 0), width - 1); };
    auto clamp_y = [height](int coord) { return min(max(coord, 0), height - 1); };

    float weight_sum = 0.0f;
    float max_w = 0.0f;
    float acc[num_planes] {}; // kept in registers (num_planes is a compile-time constant)

    for (int k = 0; k < group_count; k++) {
        int4 offset = offsets[group_start + k];
        int offset_x = offset.x;
        int offset_y = offset.y;
        int offset_t = offset.z;

        const float * buffer_bwd = buffer_bwd_base + k * frame_buffer_elems;
        const float * buffer_fwd = offset.w ? buffer_fwd_base + k * frame_buffer_elems : buffer_bwd;

        float center_weight = buffer_bwd[y * buffer_stride + x];
        float mirror_weight = buffer_fwd[clamp_y(y - offset_y) * buffer_stride + clamp_x(x - offset_x)];

        max_w = fmaxf(max_w, fmaxf(center_weight, mirror_weight));
        weight_sum += center_weight + mirror_weight;

        #pragma unroll
        for (int plane = 0; plane < num_planes; plane++) {
            auto north_west = static_cast<float>(src[((offset_t * num_planes + plane) * height + clamp_y(y + offset_y)) * image_stride + clamp_x(x + offset_x)]);
            auto south_east = static_cast<float>(src[((-offset_t * num_planes + plane) * height + clamp_y(y - offset_y)) * image_stride + clamp_x(x - offset_x)]);
            acc[plane] += center_weight * north_west + mirror_weight * south_east;
        }
    }

    int idx = y * buffer_stride + x;
    max_weight[idx] = fmaxf(max_weight[idx], max_w);
    weight[idx] += weight_sum;
    #pragma unroll
    for (int plane = 0; plane < num_planes; plane++) {
        wdst[(plane * height + y) * buffer_stride + x] += acc[plane];
    }
}

static void accumulation_grouped_dispatch(
    float * wdst,
    float * weight,
    float * max_weight,
    const void * src, int src_offset,
    const float * buffer_bwd_base,
    const float * buffer_fwd_base,
    const int4 * offsets, int group_start, int group_count,
    int frame_buffer_elems,
    int width, int height, int image_stride, int buffer_stride,
    int num_planes,
    dim3 grid, dim3 block, cudaStream_t stream,
    bool is_float, int bits_per_sample
) {

    #define LAUNCH(T, planes)                                                  \
        accumulation_grouped<T, planes><<<grid, block, 0, stream>>>(           \
            wdst, weight, max_weight,                                          \
            static_cast<const T *>(src) + src_offset,                          \
            buffer_bwd_base, buffer_fwd_base,                                  \
            offsets, group_start, group_count, frame_buffer_elems,             \
            width, height, image_stride, buffer_stride)

    if (is_float) {
        if (bits_per_sample == 32) {
            switch (num_planes) {
                case 1: LAUNCH(float, 1); break;
                case 2: LAUNCH(float, 2); break;
                case 3: LAUNCH(float, 3); break;
                default: break;
            }
        }
    } else if (bits_per_sample <= 8) {
        switch (num_planes) {
            case 1: LAUNCH(uint8_t, 1); break;
            case 2: LAUNCH(uint8_t, 2); break;
            case 3: LAUNCH(uint8_t, 3); break;
            default: break;
        }
    } else if (bits_per_sample <= 16) {
        switch (num_planes) {
            case 1: LAUNCH(uint16_t, 1); break;
            case 2: LAUNCH(uint16_t, 2); break;
            case 3: LAUNCH(uint16_t, 3); break;
            default: break;
        }
    }

    #undef LAUNCH
}

template <typename T>
__global__
__launch_bounds__(128)
static void finish(
    T * __restrict__ dst,
    const T * __restrict__ src,
    const float * __restrict__ wdst,
    const float * __restrict__ weight,
    const float * __restrict__ max_weight,
    int width, int height, int image_stride, int buffer_stride,
    int num_planes, float wref, int peak
) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return ;
    }

    auto from_float = [peak](float x) -> T {
        if constexpr (std::is_floating_point_v<T>) {
            return x;
        } else {
            return static_cast<T>(max(0, min(__float2int_rn(x), peak)));
        }
    };

    // max_weight is initialized to 0
    auto center_weight = wref * fmaxf(FLT_EPSILON, max_weight[y * buffer_stride + x]);
    auto denominator = center_weight + weight[y * buffer_stride + x];

    for (int plane = 0; plane < num_planes; plane++) {
        auto numerator = center_weight * static_cast<float>(src[(plane * height + y) * image_stride + x]) + wdst[(plane * height + y) * buffer_stride + x];
        dst[(plane * height + y) * image_stride + x] = from_float(numerator / denominator);
    }
}

static void finish_dispatch(
    void * dst,
    const void * src, int src_offset,
    const float * wdst,
    const float * weight,
    const float * max_weight,
    int width, int height, int image_stride, int buffer_stride,
    int num_planes, float wref,
    dim3 grid, dim3 block, size_t dyn_smem_size, cudaStream_t stream,
    bool is_float, int bits_per_sample
) {
    if (is_float) {
        if (bits_per_sample == 32) {
            finish<float><<<grid, block, dyn_smem_size, stream>>>(
                static_cast<float *>(dst),
                static_cast<const float *>(src) + src_offset,
                wdst,
                weight,
                max_weight,
                width, height, image_stride, buffer_stride,
                num_planes, wref, 0
            );
        }
    } else {
        int peak = (1 << bits_per_sample) - 1;

        if (bits_per_sample <= 8) {
            finish<uint8_t><<<grid, block, dyn_smem_size, stream>>>(
                static_cast<uint8_t *>(dst),
                static_cast<const uint8_t *>(src) + src_offset,
                wdst,
                weight,
                max_weight,
                width, height, image_stride, buffer_stride,
                num_planes, wref, peak
            );
        } else if (bits_per_sample <= 16) {
            finish<uint16_t><<<grid, block, dyn_smem_size, stream>>>(
                static_cast<uint16_t *>(dst),
                static_cast<const uint16_t *>(src) + src_offset,
                wdst,
                weight,
                max_weight,
                width, height, image_stride, buffer_stride,
                num_planes, wref, peak
            );
        }
    }
}

cudaError_t nlmeans(
    void * d_dst,
    void * d_src,
    float * d_buffer,
    float * d_buffer_bwd,
    float * d_buffer_fwd,
    float * d_wdst,
    float * d_weight,
    float * d_max_weight,
    const int4 * d_offsets,
    const int4 * h_offsets,
    int num_offsets,
    int group_size,
    bool is_float,
    int bits_per_sample,
    int width, int height, int image_stride, int buffer_stride,
    int radius, int block_radius, float h2_inv_norm,
    ChannelMode channels, int wmode, float wref, bool has_ref,
    cudaStream_t stream
) {

    int num_planes {};
    if (channels == ChannelMode::Y) {
        num_planes = 1;
    } else if (channels == ChannelMode::UV) {
        num_planes = 2;
    } else if (channels == ChannelMode::YUV || channels == ChannelMode::RGB) {
        num_planes = 3;
    }

    if (auto error = cudaMemsetAsync(d_wdst, 0, num_planes * height * buffer_stride * sizeof(float), stream); error != cudaSuccess) {
        return error;
    }
    if (auto error = cudaMemsetAsync(d_weight, 0, height * buffer_stride * sizeof(float), stream); error != cudaSuccess) {
        return error;
    }
    if (auto error = cudaMemsetAsync(d_max_weight, 0, height * buffer_stride * sizeof(float), stream); error != cudaSuccess) {
        return error;
    }

    dim3 block { 16, 8, 1 };
    // assert(block.x * block.y == 128);

    dim3 grid { (width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1 };

    constexpr int hrz_result = 3;
    dim3 hrz_grid { (width + hrz_result * block.x - 1) / (hrz_result * block.x), (height + block.y - 1) / block.y, 1 };
    int hrz_smem_stride = hrz_result * block.x + 2 * block_radius;
    auto hrz_smem = block.y * hrz_smem_stride * sizeof(float);

    constexpr int vrt_result = 3;
    dim3 vrt_grid { (width + block.x - 1) / block.x, (height + vrt_result * block.y - 1) / (vrt_result * block.y), 1 };
    auto vrt_smem = (vrt_result * block.y + 2 * block_radius) * block.x * sizeof(float);

    int frame_buffer_elems = height * buffer_stride;
    int center_base = (has_ref * (2 * radius + 1) + radius) * num_planes * height * image_stride;
    int src_center = radius * num_planes * height * image_stride;

    for (int group_start = 0; group_start < num_offsets; group_start += group_size) {
        int group_count = std::min(group_size, num_offsets - group_start);

        for (int k = 0; k < group_count; k++) {
            int4 offset = h_offsets[group_start + k];
            int offset_x = offset.x;
            int offset_y = offset.y;
            int offset_t = offset.z;

            distance_horizontal_dispatch<hrz_result>(
                d_buffer,
                d_src,
                center_base,
                center_base + offset_t * num_planes * height * image_stride,
                width, height, image_stride, buffer_stride,
                offset_x, offset_y,
                block_radius, channels,
                hrz_grid, block, hrz_smem, stream,
                is_float, bits_per_sample
            );

            vertical<vrt_result><<<vrt_grid, block, vrt_smem, stream>>>(
                d_buffer_bwd + k * frame_buffer_elems,
                d_buffer,
                width, height, buffer_stride,
                block_radius, h2_inv_norm, wmode
            );

            if (offset.w) {
                distance_horizontal_dispatch<hrz_result>(
                    d_buffer,
                    d_src,
                    center_base - offset_t * num_planes * height * image_stride,
                    center_base,
                    width, height, image_stride, buffer_stride,
                    offset_x, offset_y,
                    block_radius, channels,
                    hrz_grid, block, hrz_smem, stream,
                    is_float, bits_per_sample
                );

                vertical<vrt_result><<<vrt_grid, block, vrt_smem, stream>>>(
                    d_buffer_fwd + k * frame_buffer_elems,
                    d_buffer,
                    width, height, buffer_stride,
                    block_radius, h2_inv_norm, wmode
                );
            }
        }

        accumulation_grouped_dispatch(
            d_wdst, d_weight, d_max_weight,
            d_src, src_center,
            d_buffer_bwd, d_buffer_fwd,
            d_offsets, group_start, group_count,
            frame_buffer_elems,
            width, height, image_stride, buffer_stride,
            num_planes,
            grid, block, stream,
            is_float, bits_per_sample
        );
    }

    finish_dispatch(
        d_dst,
        d_src, radius * num_planes * height * image_stride,
        d_wdst, d_weight, d_max_weight,
        width, height, image_stride, buffer_stride,
        num_planes, wref,
        grid, block, 0, stream,
        is_float, bits_per_sample
    );

    return cudaSuccess;
}
