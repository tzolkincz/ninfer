#include "ops/softmax_attention/dense/causal_cache/k16v4/tiled_launch.h"
#include <stdexcept>

namespace ninfer::ops::detail {

void k16v4_kv_tiled_attention(const CausalAttentionOperands& p, K16V4KvReadView cache,
                             CausalKvPartition partition, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    (void)p;
    (void)cache;
    (void)partition;
    (void)workspace;
    (void)stream;
    throw std::logic_error("K16V4 tiled attention path not yet implemented");
}

} // namespace ninfer::ops::detail
