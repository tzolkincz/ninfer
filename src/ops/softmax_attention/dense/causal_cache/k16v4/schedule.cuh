#pragma once

#include "ops/softmax_attention/common/causal_geometry.h"
#include "ops/softmax_attention/common/mxfp8_tiled_plan.h"

namespace ninfer::ops::detail {

// BF16 K (2 bytes/element, no scale) + NVFP4 V (128B codes + 16B scales per 256 dims).
// Dynamic arena: K(BF16) + V(NVFP4 codes) + V(FP16 dequant workspace) = KeyRows * 1152 bytes.
template <int QueryRows, int KeyRows = 32, int KVWarps = 1, int MinBlocks = 2, int FixedWidth = 0>
struct K16V4KvGroupedMmaSchedule {
    static_assert(QueryRows % 16 == 0 && QueryRows >= 16 && QueryRows <= 128);
    static_assert(KeyRows == 32 || KeyRows == 64);
    static_assert(KVWarps == 1 || KVWarps == 2 || KVWarps == 4);
    static_assert(KeyRows % (16 * KVWarps) == 0);
    static_assert(QueryRows <= 2 * KeyRows);
    static_assert(FixedWidth == 0 || FixedWidth == 1);
    static constexpr int kFixedWidth = FixedWidth;
    static constexpr int kQueryRows  = QueryRows;
    static constexpr int kKeyRows    = KeyRows;
    static constexpr int kWarpsQ     = QueryRows / 16;
    static constexpr int kWarpsKV    = KVWarps;
    static constexpr int kWarps      = kWarpsQ * kWarpsKV;
    static_assert(kWarps <= 8);
    static constexpr int kThreads            = 32 * kWarps;
    static constexpr int kLaunchBoundThreads = kThreads < 128 ? 128 : kThreads;
    static constexpr int kMinBlocks          = MinBlocks;
    static constexpr int kArenaBytes         = KeyRows * 256 * 9 / 2;
};

// Tiled: BF16 Q/K + NVFP4 V dequant.
template <int QueryTile = kMxfp8TiledQueryRows, int KeyTile = 64, int MaxRegisters = 255>
struct K16V4KvTiledMmaSchedule {
    static_assert(QueryTile == 16 || QueryTile == 32 || QueryTile == 64 || QueryTile == 128);
    static_assert(KeyTile == 32 || KeyTile == 64);
    static_assert(MaxRegisters > 0 && MaxRegisters <= 255);
    static constexpr int kQueryRows    = QueryTile;
    static constexpr int kKeyRows      = KeyTile;
    static constexpr int kRowTiles     = QueryTile / 16;
    static constexpr int kWarps        = kRowTiles;
    static constexpr int kThreads      = kWarps * 32;
    static constexpr int kMaxRegisters = MaxRegisters;
    static constexpr int kQBytes       = QueryTile * 256 * 2;
    static constexpr int kKBytes       = KeyTile * 256 * 2;
    static constexpr int kVBytes       = KeyTile * 128;
    static constexpr int kVStageBytes  = KeyTile * 256 * 2;
    static constexpr int kScaleBytes   = KeyTile * 16;
    static constexpr int kSharedBytes =
        kQBytes + kKBytes + kVBytes + kVStageBytes + kScaleBytes;
    static_assert(kSharedBytes <= 99 * 1024);
};

struct K16V4KvMergeSchedule {
    static constexpr int kDChunk  = 256;
    static constexpr int kThreads = 256;
};

} // namespace ninfer::ops::detail
