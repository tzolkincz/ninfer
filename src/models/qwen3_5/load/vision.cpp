#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5::loading {

VisionWeights bind_vision(Bindings& b, const VisionConfig& config, const TextConfig& target) {
    // The vision tower is host-resident: its weights live in pinned host RAM so they do not
    // occupy the vision rank's VRAM. Each encode re-reads them over the interconnect, and that
    // traffic hides behind the tower's compute.
    struct Host {
        explicit Host(Bindings& b) : b(b) {}
        WeightId parameter(std::string name, artifact::Shape shape, std::vector<std::string> inputs) {
            return b.parameter(std::move(name), std::move(shape), std::move(inputs), {},
                               artifact::Residency::Host);
        }
        WeightId direct(std::string name, artifact::Shape shape, QType format = QType::BF16) {
            return b.direct(std::move(name), std::move(shape), format, artifact::Residency::Host);
        }
        Bindings& b;
    } host(b);

    const auto h            = config.hidden_size;
    const auto intermediate = config.intermediate_size;
    VisionWeights out;
    out.patch_embedding =
        host.parameter("vision/patch_embedding", {h, config.patch_width()}, {"vision/patch_input"});
    out.patch_embedding_bias = host.direct("vision/patch_embedding_bias", {h});
    out.position_embedding =
        host.direct("vision/position_embedding", {config.num_position_embeddings, h});
    out.layers.reserve(config.depth);
    for (std::uint32_t i = 0; i < config.depth; ++i) {
        const auto p = "vision/layers/" + std::to_string(i) + "/";
        VisionBlockWeights layer;
        layer.norm1       = {host.direct(p + "norm1_weight", {h}),
                             host.direct(p + "norm1_bias", {h})};
        layer.norm2       = {host.direct(p + "norm2_weight", {h}),
                             host.direct(p + "norm2_bias", {h})};
        layer.query       = host.parameter(p + "attention/query", {h, h}, {p + "attention_input"});
        layer.key         = host.parameter(p + "attention/key", {h, h}, {p + "attention_input"});
        layer.value       = host.parameter(p + "attention/value", {h, h}, {p + "attention_input"});
        layer.query_bias  = host.direct(p + "attention/query_bias", {h});
        layer.key_bias    = host.direct(p + "attention/key_bias", {h});
        layer.value_bias  = host.direct(p + "attention/value_bias", {h});
        layer.output      = host.parameter(p + "attention/output", {h, h}, {p + "attention_output"});
        layer.output_bias = host.direct(p + "attention/output_bias", {h});
        layer.fc1         = host.parameter(p + "mlp/fc1", {intermediate, h}, {p + "mlp_input"});
        layer.fc1_bias    = host.direct(p + "mlp/fc1_bias", {intermediate});
        layer.fc2         = host.parameter(p + "mlp/fc2", {h, intermediate}, {p + "mlp_activation"});
        layer.fc2_bias    = host.direct(p + "mlp/fc2_bias", {h});
        out.layers.push_back(layer);
    }
    const auto merger = config.merger_width();
    out.merger_norm   = {host.direct("vision/merger/norm_weight", {h}),
                         host.direct("vision/merger/norm_bias", {h})};
    out.merger_fc1    = host.parameter("vision/merger/fc1", {merger, merger}, {"vision/merger/input"});
    out.merger_fc1_bias = host.direct("vision/merger/fc1_bias", {merger});
    out.merger_fc2      = host.parameter("vision/merger/fc2", {target.hidden_size, merger},
                                         {"vision/merger/activation"});
    out.merger_fc2_bias = host.direct("vision/merger/fc2_bias", {target.hidden_size});
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
