#include "ops/linear/q5/q5_shapes.h"

namespace ninfer::ops::detail {
Q5Launch select_q5_n12288_k5120(std::int32_t tokens) {
    // 48 row blocks exceed the r16/r32 sliced instances, so T>=3 uses the r64 MMA tile
    // (padded to 64 blocks); the T=1/2 direct SIMT routes are row-generic.
    if (tokens <= 1) return launch_q5_a16_direct_r1_t1_w4_k5120;
    if (tokens <= 2) return launch_q5_a16_direct_r1_t2_w4_k5120;
    return launch_q5_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
