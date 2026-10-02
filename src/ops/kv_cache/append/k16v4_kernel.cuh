#pragma once

// K16V4 append kernel: K stays BF16 (no rotation/quantization); V uses NVFP4 (Hadamard + quantize).

#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"

#include "ops/kv_cache/nvfp4_group16_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

template <class Geometry>
__device__ __forceinline__ std::int64_t k16v4_kv_cache_k_index(int page, int head, int d,
                                                               int page_offset) {
    return paged_kv_element_offset<kKVCacheAppendFullHeadDim, Geometry::KVHeads>(
        page, head, page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ void kv_cache_append_full_k16v4_row(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    std::uint8_t* __restrict__ scale_v, int token, int kv_head, int physical_page,
    int page_offset, int lane, float* scratch) {
    constexpr unsigned FullMask = 0xffffffffU;
    // K: BF16 copy (no rotation, no quantization)
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        cache_k[k16v4_kv_cache_k_index<Geometry>(physical_page, kv_head, d, page_offset)] =
            k[kv_cache_nvfp4_src_index<Geometry>(kv_head, d, token)];
    }
    // V: NVFP4 (Hadamard + quantize)
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        values[r] = __bfloat162float(v[kv_cache_nvfp4_src_index<Geometry>(kv_head, d, token)]);
    }
    // Raw NVFP4 encoding (no Hadamard rotation)
#pragma unroll
    for (int r = 0; r < 8; ++r) scratch[lane + 32 * r] = values[r];
    __syncwarp();
    if (lane < kKVCacheNvfp4Groups) {
        const auto quantized = kv_cache_nvfp4_quantize_group16(scratch + lane * kKVCacheNvfp4Group);
        const std::int64_t code_offset = kv_cache_nvfp4_code_index<Geometry>(
            physical_page, kv_head, lane * kKVCacheNvfp4Group, page_offset);
        store_vec(cache_v + code_offset, make_uint2(quantized.codes_lo, quantized.codes_hi));
        scale_v[kv_cache_nvfp4_scale_index<Geometry>(physical_page, kv_head, lane, page_offset)] =
            quantized.scale;
    }
}

template <typename Geometry, typename Metadata, bool MultiBatch = false>
__launch_bounds__(256) __global__
    void kv_cache_append_full_k16v4_kernel(const __nv_bfloat16* __restrict__ k,
                                          const __nv_bfloat16* __restrict__ v,
                                          const std::int32_t* __restrict__ positions,
                                          Metadata metadata, __nv_bfloat16* __restrict__ cache_k,
                                          std::uint8_t* __restrict__ cache_v,
                                          std::uint8_t* __restrict__ scale_v,
                                          std::int32_t width) {
    if constexpr (MultiBatch) {
        const int batch = blockIdx.z;
        metadata.table_rows += batch;
        if (metadata.valid_columns) metadata.valid_columns += batch;
        const auto offset = static_cast<std::int64_t>(batch) * width * 256 * Geometry::KVHeads;
        k += offset;
        v += offset;
        positions += batch * width;
    }
    constexpr int Warps = 8;
    __shared__ float scratch[Warps][kKVCacheNvfp4HeadDim];
    const int tokens = metadata.valid_tokens(width);
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int unit   = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units  = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page = __shfl_sync(0xffffffffU, physical_page, 0);
    kv_cache_append_full_k16v4_row<Geometry>(k, v, cache_k, cache_v, scale_v, token, kv_head,
                                             physical_page, position & kPagedKVPageMask, lane,
                                             scratch[warp]);
}

} // namespace ninfer::ops
