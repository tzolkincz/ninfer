#pragma once

#include "ops/softmax_attention/common/causal_geometry.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/common/causal_tile_io.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/k16v4/operands.h"
#include "ops/softmax_attention/dense/causal_cache/k16v4/schedule.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/epilogue.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/softmax.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/split_policy.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops::detail {

template <class G, class S, bool MultiBatch, bool Masked, class Input>
__launch_bounds__(S::kLaunchBoundThreads, S::kMinBlocks) __global__
    void k16v4_kv_grouped_mma_kernel(const __nv_bfloat16* q, Input input, const int* positions,
                                     typename K16V4KvCacheView<Input::writes_cache>::Key* cache_k,
                                     typename K16V4KvCacheView<Input::writes_cache>::VCode* cache_v,
                                     typename K16V4KvCacheView<Input::writes_cache>::VScale* cache_v_scale,
                                     const int* tables,
                                     const int* validity, const int* table_rows, int table_stride,
                                     int runtime_width, float scale, CausalKvPartition partition,
                                     CausalPartialView partial) {
    const int width = runtime_width;
    constexpr int D = G::kHeadDim;
    constexpr int M = S::kQueryRows;
    constexpr int N = S::kKeyRows;
    constexpr int WQ = S::kWarpsQ;
    constexpr int WK = S::kWarpsKV;
    constexpr int NK = N / WK;
    constexpr int PStride = (N == 32) ? 64 : N;
    constexpr int QKNt = NK / 8;
    constexpr int QKKs = D / 16;
    constexpr int PVNt = D / 8;
    constexpr int PVKs = NK / 16;

    __shared__ __align__(16) __nv_bfloat16 q_s[M * D];
    __shared__ __align__(16) __half p_s[M * PStride];
    __shared__ std::uint8_t v_scale_s[N * kKVCacheNvfp4Groups];
    __shared__ float smax[WK * M];
    __shared__ float ssum[WK * M];
    __shared__ float reduction[WK > 1 ? WK * 16 * D : 1];

    extern __shared__ __align__(16) unsigned char arena[];
    auto* k_s     = reinterpret_cast<__nv_bfloat16*>(arena);
    auto* v_codes = reinterpret_cast<std::uint8_t*>(arena + N * D * 2);
    auto* v_f16   = reinterpret_cast<__half*>(arena + N * D * 2 + N * D / 2);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int warp_q = warp % WQ, warp_kv = warp / WQ;
    const int head = S::kFixedWidth ? blockIdx.x : blockIdx.x % G::KVHeads;
    const int tile = S::kFixedWidth ? 0 : blockIdx.x / G::KVHeads;
    const int split = blockIdx.y, batch = MultiBatch ? blockIdx.z : 0;
    const int row_begin = tile * M, packed_rows = width * G::GroupSize;
    const int live = Masked ? validity[batch] : width;
    q += static_cast<std::int64_t>(batch) * width * D * G::QHeads;
    positions += batch * width;
    if constexpr (Input::writes_cache) {
        input.k += static_cast<std::int64_t>(batch) * width * D * G::KVHeads;
        input.v += static_cast<std::int64_t>(batch) * width * D * G::KVHeads;
    }
    partial.acc += static_cast<std::int64_t>(batch) * width * G::QHeads * D * partition.capacity;
    partial.maximum += static_cast<std::int64_t>(batch) * width * G::QHeads * partition.capacity;
    partial.sum += static_cast<std::int64_t>(batch) * width * G::QHeads * partition.capacity;
    const auto neutral = [&]() {
        for (int i = tid; i < M * (D / 2); i += S::kThreads) {
            const int row = row_begin + i / (D / 2), d = i % (D / 2) * 2;
            if (row < packed_rows) {
                const int token = row / G::GroupSize;
                const int h     = head * G::GroupSize + row % G::GroupSize;
                bf16_kv_store_pair<G, true>(partial, nullptr, h, token, d, width, split, 0.0f, 0.0f,
                                            -CUDART_INF_F, 0.0f);
            }
        }
    };
    if (row_begin >= live * G::GroupSize) {
        neutral();
        return;
    }
    const int window = positions[live - 1] + 1;
    const int active = partition.active(window);
    if (split >= active) return;
    const int logical_tiles = div_up(window, N);
    const int start = (split * logical_tiles / active) * N;
    const int stop  = min(window, ((split + 1) * logical_tiles / active) * N);
    const int table_row = table_rows ? table_rows[batch] : 0;
    const int* table    = tables + static_cast<std::int64_t>(table_row) * table_stride;

    // Append: K as BF16 (no rotation), V as NVFP4 (Hadamard + quantize)
    if constexpr (Input::writes_cache) {
        if (tile == 0) {
            for (int i = tid; i < live * (D / 8); i += S::kThreads) {
                const int token = i / (D / 8), d = i % (D / 8) * 8, key = positions[token];
                if (key >= start && key < stop) {
                    const auto src = causal_new_index<G>(head, d, token);
                    const auto dst = bf16_kv_cache_index<G>(table[key >> kPagedKVPageShift], head,
                                                            d, key & kPagedKVPageMask);
                    store_vec(cache_k + dst, load_vec<int4>(input.k + src));
                }
            }
            for (int i = tid; i < live * (D / 16); i += S::kThreads) {
                const int token = i / (D / 16), d16 = i % (D / 16) * 16, key = positions[token];
                if (key >= start && key < stop) {
                    float values[16];
                    for (int r = 0; r < 8; ++r) {
                        const auto idx = causal_new_index<G>(head, d16 + 2 * r, token);
                        auto bits = load_vec<std::uint32_t>(input.v + idx);
                        __nv_bfloat162 h2 = *reinterpret_cast<__nv_bfloat162*>(&bits);
                        values[2 * r]     = __bfloat162float(h2.x);
                        values[2 * r + 1] = __bfloat162float(h2.y);
                    }
                    // Raw NVFP4 encoding (no Hadamard rotation)
                    auto qz = kv_cache_nvfp4_quantize_group16(values);
                    const int page = table[key >> kPagedKVPageShift], poff = key & kPagedKVPageMask;
                    const int64_t co = kv_cache_nvfp4_code_index<G>(page, head, d16, poff);
                    *reinterpret_cast<std::uint32_t*>(&cache_v[co]) = qz.codes_lo;
                    *reinterpret_cast<std::uint32_t*>(&cache_v[co + 4]) = qz.codes_hi;
                    const int64_t so = kv_cache_nvfp4_scale_index<G>(page, head, d16 / 16, poff);
                    cache_v_scale[so] = qz.scale;
                }
            }
        }
    }

    const int last_token = min(live, div_up(row_begin + M, G::GroupSize)) - 1;
    const int end        = min(stop, positions[last_token] + 1);
    if (start >= end) {
        neutral();
        return;
    }

    // Load Q (BF16, no rotation)
    for (int i = tid; i < M * (D / 8); i += S::kThreads) {
        const int row = i / (D / 8), d = i % (D / 8) * 8, packed = row_begin + row;
        const int token = packed / G::GroupSize, h = head * G::GroupSize + packed % G::GroupSize;
        auto* dst = &q_s[row * D + causal_swizzle(row, d)];
        if (token < live)
            cp_async<16>(dst, q + causal_q_index<G>(h, d, token));
        else
            store_vec(dst, make_int4(0, 0, 0, 0));
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    unsigned q_frag[QKKs][4];
    {
        const int a_row = warp_q * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int d = k * 16 + (lane >> 4) * 8;
            ldmatrix_x4(q_frag[k][0], q_frag[k][1], q_frag[k][2], q_frag[k][3],
                        smem_addr(&q_s[a_row * D + causal_swizzle(a_row, d)]));
        }
    }
    __syncthreads();

    float acc[PVNt][4] = {};
    Bf16KvSoftmaxRow state[2];
    const float scale_log2 = scale * kLog2E;
    const int gid = lane >> 2, lid = lane & 3;
    const int rows[2]   = {row_begin + warp_q * 16 + gid, row_begin + warp_q * 16 + gid + 8};
    const int tokens[2] = {rows[0] / G::GroupSize, rows[1] / G::GroupSize};
    const int qabs[2]   = {tokens[0] < live ? positions[tokens[0]] : -1,
                           tokens[1] < live ? positions[tokens[1]] : -1};
    for (int k0 = start; k0 < end; k0 += N) {
        {
            // K: BF16
#pragma unroll 1
            for (int i = tid; i < N * (D / 8); i += S::kThreads) {
                const int row = i / (D / 8), d = i % (D / 8) * 8, key = k0 + row;
                auto* kd = &k_s[row * D + causal_swizzle(row, d)];
                if (key < end) {
                    const int physical_page = table[key >> kPagedKVPageShift];
                    const int ko = bf16_kv_cache_index<G>(physical_page, head, d,
                                                          key & kPagedKVPageMask);
                    cp_async<16>(kd, cache_k + ko);
                } else
                    store_vec(kd, make_int4(0, 0, 0, 0));
            }
            // V codes: NVFP4 (4 bytes = 8 elements)
#pragma unroll 1
            for (int i = tid; i < N * (D / 8); i += S::kThreads) {
                const int row = i / (D / 8), d8 = i % (D / 8) * 4, key = k0 + row;
                if (key < end) {
                    const int physical_page = table[key >> kPagedKVPageShift];
                    const int vo = kv_cache_nvfp4_code_index<G>(physical_page, head, d8 * 2,
                                                                key & kPagedKVPageMask);
                    cp_async<4>(&v_codes[row * (D / 2) + d8], cache_v + vo);
                } else {
                    *reinterpret_cast<int*>(&v_codes[row * (D / 2) + d8]) = 0;
                }
            }
            // V scales
#pragma unroll 1
            for (int i = tid; i < N * kKVCacheNvfp4Groups; i += S::kThreads) {
                const int row = i / kKVCacheNvfp4Groups, grp = i % kKVCacheNvfp4Groups,
                          key = k0 + row;
                if (key < end) {
                    const int physical_page = table[key >> kPagedKVPageShift];
                    const int so = kv_cache_nvfp4_scale_index<G>(physical_page, head, grp,
                                                                 key & kPagedKVPageMask);
                    v_scale_s[row * kKVCacheNvfp4Groups + grp] = cache_v_scale[so];
                } else
                    v_scale_s[row * kKVCacheNvfp4Groups + grp] = 0;
            }
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();

        // Dequant V
        for (int i = tid; i < N * (D / 8); i += S::kThreads) {
            const int row = i / (D / 8), d8 = i % (D / 8) * 8, key = k0 + row;
            const int grp = d8 / 16;
            if (key < end) {
                const int4 deq = kv_cache_nvfp4_dequant_f16x8(
                    &v_codes[row * (D / 2) + d8 / 2], v_scale_s[row * kKVCacheNvfp4Groups + grp]);
                store_vec(&v_f16[row * D + causal_swizzle(row, d8)], deq);
            } else
                store_vec(&v_f16[row * D + causal_swizzle(row, d8)], make_int4(0, 0, 0, 0));
        }
        __syncthreads();

        // QK^T: BF16 MMA
        float score[QKNt][4] = {};
#pragma unroll
        for (int n = 0; n < QKNt; ++n) {
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned bf[2];
                const int row = warp_kv * NK + n * 8 + (lane & 7);
                const int d   = k * 16 + ((lane >> 3) & 1) * 8;
                ldmatrix_x2(bf[0], bf[1], smem_addr(&k_s[row * D + causal_swizzle(row, d)]));
                mma_bf16(score[n][0], score[n][1], score[n][2], score[n][3], q_frag[k][0],
                         q_frag[k][1], q_frag[k][2], q_frag[k][3], bf[0], bf[1]);
            }
        }
        float maximum[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
        for (int n = 0; n < QKNt; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int key = k0 + warp_kv * NK + n * 8 + 2 * lid + (j & 1);
                if (key >= end || key > qabs[j / 2]) score[n][j] = -CUDART_INF_F;
                maximum[j / 2] = fmaxf(maximum[j / 2], score[n][j]);
            }
        }
        const float alpha[2] = {state[0].update(warp_max<4>(maximum[0], 0xffffffffu), scale_log2),
                                state[1].update(warp_max<4>(maximum[1], 0xffffffffu), scale_log2)};
        unsigned pf[PVKs][4];
        float tile_sum[2] = {};
#pragma unroll
        for (int n = 0; n < QKNt; ++n) {
            const float p0 = state[0].probability(score[n][0], scale_log2);
            const float p1 = state[0].probability(score[n][1], scale_log2);
            const float p2 = state[1].probability(score[n][2], scale_log2);
            const float p3 = state[1].probability(score[n][3], scale_log2);
            tile_sum[0] += p0 + p1;
            tile_sum[1] += p2 + p3;
            pf[n / 2][(n % 2) * 2]     = pack_f16x2(p0, p1);
            pf[n / 2][(n % 2) * 2 + 1] = pack_f16x2(p2, p3);
        }
        state[0].accumulate(alpha[0], tile_sum[0]);
        state[1].accumulate(alpha[1], tile_sum[1]);
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha[0];
            acc[n][1] *= alpha[0];
            acc[n][2] *= alpha[1];
            acc[n][3] *= alpha[1];
        }
        // PV: FP16 MMA
        if constexpr (WQ >= 4 && (MultiBatch || !Input::writes_cache)) {
            unsigned vf[2][4];
            const auto load_v = [&](int i, int slot) {
                const int k = i / (PVNt / 2), n = i % (PVNt / 2) * 2;
                const int row = warp_kv * NK + k * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                const int d   = n * 8 + (lane >> 4) * 8;
                ldmatrix_x4_t(vf[slot][0], vf[slot][1], vf[slot][2], vf[slot][3],
                              smem_addr(&v_f16[row * D + causal_swizzle(row, d)]));
            };
            load_v(0, 0);
#pragma unroll
            for (int i = 0; i < PVKs * (PVNt / 2); ++i) {
                const int k = i / (PVNt / 2), n = i % (PVNt / 2) * 2, slot = i & 1;
                if (i + 1 < PVKs * (PVNt / 2)) load_v(i + 1, slot ^ 1);
                mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[k][0], pf[k][1], pf[k][2],
                        pf[k][3], vf[slot][0], vf[slot][1]);
                mma_f16(acc[n + 1][0], acc[n + 1][1], acc[n + 1][2], acc[n + 1][3], pf[k][0],
                        pf[k][1], pf[k][2], pf[k][3], vf[slot][2], vf[slot][3]);
            }
        } else {
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
#pragma unroll
                for (int k = 0; k < PVKs; ++k) {
                    unsigned vf[2];
                    const int row = warp_kv * NK + k * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                    const int d   = n * 8;
                    ldmatrix_x2_t(vf[0], vf[1],
                                  smem_addr(&v_f16[row * D + causal_swizzle(row, d)]));
                    mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[k][0], pf[k][1],
                            pf[k][2], pf[k][3], vf[0], vf[1]);
                }
            }
        }
        __syncthreads();
    }

    state[0].finish();
    state[1].finish();
    float m[2]    = {state[0].maximum * scale_log2, state[1].maximum * scale_log2};
    float sums[2] = {state[0].sum, state[1].sum};
    if constexpr (WK > 1) {
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = warp_kv * 16 + gid + j * 8;
            if (lid == 0) {
                smax[row] = m[j];
                ssum[row] = sums[j];
            }
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                const int d = n * 8 + 2 * lid;
                *reinterpret_cast<float2*>(&reduction[row * D + d]) =
                    make_float2(acc[n][j * 2], acc[n][j * 2 + 1]);
            }
        }
        __syncthreads();
        if (warp_kv != 0) return;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int row = gid + j * 8;
            m[j]          = -CUDART_INF_F;
            sums[j]       = 0;
#pragma unroll
            for (int w = 0; w < WK; ++w) m[j] = fmaxf(m[j], smax[w * 16 + row]);
            float weights[WK];
#pragma unroll
            for (int w = 0; w < WK; ++w) {
                weights[w] = bf16_kv_state_weight(smax[w * 16 + row], ssum[w * 16 + row], m[j]);
                sums[j] += weights[w] * ssum[w * 16 + row];
            }
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                float a = 0, b = 0;
#pragma unroll
                for (int w = 0; w < WK; ++w) {
                    const auto x = *reinterpret_cast<const float2*>(
                        &reduction[(w * 16 + row) * D + n * 8 + 2 * lid]);
                    a += weights[w] * x.x;
                    b += weights[w] * x.y;
                }
                acc[n][2 * j]     = a;
                acc[n][2 * j + 1] = b;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        if (tokens[j] < width) {
            const int h = head * G::GroupSize + rows[j] % G::GroupSize;
#pragma unroll
            for (int n = 0; n < PVNt; ++n)
                bf16_kv_store_pair<G, true>(partial, nullptr, h, tokens[j], n * 8 + 2 * lid, width,
                                            split, acc[n][2 * j], acc[n][2 * j + 1], m[j], sums[j]);
        }
    }
}

} // namespace ninfer::ops::detail
