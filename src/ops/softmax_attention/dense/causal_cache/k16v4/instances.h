#pragma once

#include "ops/softmax_attention/dense/causal_cache/k16v4/schedule.cuh"

namespace ninfer::ops::detail {

template <class G, int Tokens>
struct K16V4KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles = (Tokens * G::GroupSize + 15) / 16;
    static constexpr int kBr       = kRowTiles * 16;
    // WK=2 only for M=16 (decode) to keep reduction buffer within shared memory budget.
    static constexpr int kKVWarps  = (Tokens == 1 && kBr == 16) ? 2 : 1;
    using Schedule                 = K16V4KvGroupedMmaSchedule<
                                              kBr, kRowTiles >= 3 ? 32 : (Tokens == 1 ? 32 : 64),
                                              kKVWarps, Tokens == 1 ? 2 : 1>;
    using Merge                    = K16V4KvMergeSchedule;
};

using K16V4KvTiledInstance = K16V4KvTiledMmaSchedule<kMxfp8TiledQueryRows, 64>;

} // namespace ninfer::ops::detail
