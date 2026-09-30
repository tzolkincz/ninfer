#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/fp8/fp8_a16_simt.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {

template <int K>
void launch_matrix(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    const auto operands = fp8_a16_operands(x, weight);
    auto* data          = static_cast<__nv_bfloat16*>(residual.data);
    const LinearBf16Output output{data, weight.n};
    const LinearResidualAddEpilogue epilogue{{data, weight.n}};
    // The short-K residual projection retains its lower-latency SIMT decode capacities.
    if constexpr (K == 6144) {
        if (x.ne[1] <= 4) {
            const auto tiny = [&]<int Tokens>() {
                using Schedule =
                    Fp8A16SimtSchedule<8, 1, 16, Tokens, 1, Fp8SimtActivationAccess::TokenPacked,
                                       Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous,
                                       1>;
                launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, K, Tokens, true>>(
                    operands, output, epilogue, stream);
            };
            if (x.ne[1] == 2) return tiny.template operator()<2>();
            if (x.ne[1] == 3) return tiny.template operator()<3>();
            return tiny.template operator()<4>();
        }
    }
    const auto sliced = [&]<int T, int W, int Stages>() {
        launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Fp8SlicedInstance<T, W, Stages>, K>>(
            operands, output, epilogue, stream);
    };
    if constexpr (K == 17408) {
        if (x.ne[1] <= 8) return sliced.template operator()<8, 8, 2>();
    }
    if (x.ne[1] <= 16) return sliced.template operator()<16, 8, 2>();
    if constexpr (K == Fp8N5120K6144::kInputRows || K == Fp8N5120K3072::kInputRows) {
        if (x.ne[1] <= 32) return sliced.template operator()<16, 4, 2>();
        if (x.ne[1] <= 64) return sliced.template operator()<32, 4, 1>();
    } else {
        if (x.ne[1] <= 32) return sliced.template operator()<32, 8, 1>();
        if (x.ne[1] <= 64) return sliced.template operator()<64, 2, 2>();
    }
    if (x.ne[1] <= 128)
        return launch_fp8_a16_mma<
            Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 64, 128, 32, 16, 2, 2>, K>>(operands, output,
                                                                                  epilogue, stream);
    launch_fp8_a16_mma<Fp8ScheduleInstance<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, K>>(
        operands, output, epilogue, stream);
}

void fp8_linear_add_matrix_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  cudaStream_t stream) {
    switch (resolve_fp8_geometry(weight.n, weight.k)) {
    case Fp8GeometryId::N5120K6144:
        return launch_matrix<Fp8N5120K6144::kInputRows>(x, weight, residual, stream);
    case Fp8GeometryId::N5120K17408:
        return launch_matrix<Fp8N5120K17408::kInputRows>(x, weight, residual, stream);
    case Fp8GeometryId::N5120K3072:
        return launch_matrix<Fp8N5120K3072::kInputRows>(x, weight, residual, stream);
    case Fp8GeometryId::N5120K8704:
        return launch_matrix<Fp8N5120K8704::kInputRows>(x, weight, residual, stream);
    case Fp8GeometryId::N14336K5120:
    case Fp8GeometryId::N16384K5120:
    case Fp8GeometryId::N34816K5120:
    case Fp8GeometryId::N248320K5120:
    case Fp8GeometryId::N7168K5120:
    case Fp8GeometryId::N8192K5120:
    case Fp8GeometryId::N17408K5120:
    case Fp8GeometryId::N124160K5120:
        break;
    }
    throw std::invalid_argument("fp8 linear_add matrix: unsupported problem");
}
} // namespace ninfer::ops::detail
