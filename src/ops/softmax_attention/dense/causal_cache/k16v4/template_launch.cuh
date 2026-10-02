#pragma once

#include "core/device.h"
#include "ops/softmax_attention/dense/causal_cache/k16v4/operands.h"
#include "ops/softmax_attention/dense/causal_cache/k16v4/grouped_mma.cuh"
#include "ops/softmax_attention/common/causal_merge.cuh"
#include "ops/launcher/kernel_attr_once.h"
#include <stdexcept>

namespace ninfer::ops::detail {

template <class G, bool Writable>
void validate_k16v4_kv_operands(const CausalAttentionOperands& p,
                               const K16V4KvCacheView<Writable>& cache) {
    const auto aligned = [](const void* pointer) {
        return pointer && reinterpret_cast<std::uintptr_t>(pointer) % 16 == 0;
    };
    if (p.head_dim != G::kHeadDim || p.query_heads != G::QHeads || cache.kv_heads != G::KVHeads ||
        !aligned(p.q) || !aligned(p.out) || !aligned(cache.keys) || !aligned(cache.values) ||
        !aligned(cache.value_scales) || !p.positions || !cache.tables || p.width <= 0 ||
        p.batch < 1 || p.batch > 65535 || cache.table_stride <= 0 || p.visible_capacity <= 0 ||
        static_cast<std::int64_t>(p.visible_capacity) >
            static_cast<std::int64_t>(cache.table_stride) * kPagedKVPageSize ||
        (cache.valid_columns && !cache.table_rows))
        throw std::invalid_argument("K16V4 KV template operands do not match its geometry/layout");
}

template <int Bytes, auto Kernel>
int k16v4_kv_dynamic_shared() {
    static_assert(Bytes <= 99 * 1024);
    // Static shared memory (q_s, p_s, v_scale_s, pages, smax, ssum, reduction) can push
    // the total over 48KB even when the dynamic arena alone is under 48KB. Always opt in.
    static FuncAttrPerDevice attr;
    attr.ensure(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes);
    return Bytes;
}

template <class G, class S, bool MultiBatch, bool Masked, bool Writable, class Input>
void launch_k16v4_kv_grouped_mma(const CausalAttentionOperands& p, K16V4KvCacheView<Writable> cache,
                                 Input input, CausalKvPartition partition, CausalPartialView partial,
                                 cudaStream_t stream) {
    static_assert(Writable == Input::writes_cache);
    validate_k16v4_kv_operands<G>(p, cache);
    if (MultiBatch != (p.batch > 1) ||
        Masked != (cache.valid_columns != nullptr) || partition.capacity < 1 ||
        partition.target > CausalKvPartition::kMaxSplits || partition.target < 1 ||
        partition.capacity != partition.active(p.visible_capacity) || !partial.acc ||
        !partial.maximum || !partial.sum)
        throw std::invalid_argument("K16V4 grouped attention: invalid schedule/partials");
    if constexpr (Input::writes_cache)
        if (!input.k || !input.v) throw std::invalid_argument("K16V4 append requires K/V");
    constexpr auto kernel = k16v4_kv_grouped_mma_kernel<G, S, MultiBatch, Masked, Input>;
    const int bytes = k16v4_kv_dynamic_shared<S::kArenaBytes, kernel>();
    const dim3 grid(G::KVHeads * div_up(p.width * G::GroupSize, S::kQueryRows), partition.capacity,
                    p.batch);
    kernel<<<grid, S::kThreads, bytes, stream>>>(
        p.q, input, p.positions, cache.keys, cache.values, cache.value_scales, cache.tables,
        cache.valid_columns, cache.table_rows, cache.table_stride, p.width, p.scale, partition,
        partial);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
