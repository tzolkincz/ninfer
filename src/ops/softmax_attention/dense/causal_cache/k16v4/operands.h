#pragma once
#include "ops/softmax_attention/common/causal_operands.h"

namespace ninfer::ops::detail {
// K16V4: BF16 keys (no scale plane) + NVFP4 values (U8 codes + U8 scales).
template <bool Writable>
struct K16V4KvCacheView {
    using Key     = std::conditional_t<Writable, __nv_bfloat16, const __nv_bfloat16>;
    using VCode   = std::conditional_t<Writable, std::uint8_t, const std::uint8_t>;
    using VScale  = std::conditional_t<Writable, std::uint8_t, const std::uint8_t>;
    Key* keys;
    VCode* values;
    VScale* value_scales;
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    int table_stride, kv_heads;
};
using K16V4KvReadView  = K16V4KvCacheView<false>;
using K16V4KvWriteView = K16V4KvCacheView<true>;

template <bool Writable>
K16V4KvCacheView<Writable> make_k16v4_kv_cache_view(const PagedKVBatchLayerView& cache,
                                                   const Tensor* valid = nullptr,
                                                   const Tensor* rows  = nullptr) {
    return {static_cast<typename K16V4KvCacheView<Writable>::Key*>(cache.k_pages.data),
            static_cast<typename K16V4KvCacheView<Writable>::VCode*>(cache.v_pages.data),
            static_cast<typename K16V4KvCacheView<Writable>::VScale*>(cache.v_scale_pages.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            valid ? static_cast<const std::int32_t*>(valid->data) : nullptr,
            rows ? static_cast<const std::int32_t*>(rows->data) : nullptr,
            cache.block_tables.ne[0],
            cache.num_kv_heads};
}

} // namespace ninfer::ops::detail
