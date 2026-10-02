#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_partition.h"

namespace ninfer::ops::detail {

enum class K16V4KvFamily { Grouped, ParallelGrouped, Tiled };

struct K16V4KvCausalPlan {
    static constexpr int kTokenTile = 8;
    K16V4KvFamily family;
    int query_heads, width, batch, query_tile;
    CausalAttentionExecutionEnvelope envelope;
    CausalKvPartition partition;
};

K16V4KvCausalPlan make_k16v4_kv_causal_plan(int heads, int width, int batch,
                                            CausalAttentionExecutionEnvelope envelope);
std::size_t k16v4_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                     CausalAttentionExecutionEnvelope envelope);

} // namespace ninfer::ops::detail
