#include "models/qwen3_5/frontend/output_session.h"
#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "text/unicode.h"
#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5 {
namespace {
namespace fi                                = frontend;
constexpr std::string_view kThinkClose      = "</think>";
constexpr std::string_view kToolOpen        = "<tool_call>";
constexpr std::string_view kUtf8Replacement = "\xef\xbf\xbd";

std::size_t channel_index(OutputChannel channel) noexcept {
    return channel == OutputChannel::Reasoning ? 0 : 1;
}

void append_delta(PublishedOutput& output, OutputChannel channel, std::string text) {
    if (text.empty()) { return; }
    if (!output.empty() && output.back().channel == channel) {
        output.back().text += text;
    } else {
        output.push_back(OutputDelta{.channel = channel, .text = std::move(text)});
    }
}

std::string consume_generated_utf8(std::string& pending) {
    std::string decoded;
    decoded.reserve(pending.size());
    std::size_t offset = 0;
    while (offset < pending.size()) {
        const auto lead    = static_cast<unsigned char>(pending[offset]);
        std::size_t length = 0;
        if (lead <= 0x7fU) {
            decoded.push_back(pending[offset]);
            ++offset;
            continue;
        } else if (lead >= 0xc2U && lead <= 0xdfU) {
            length = 2;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            length = 3;
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            length = 4;
        } else {
            decoded.append(kUtf8Replacement);
            ++offset;
            continue;
        }

        bool malformed = false;
        for (std::size_t index = 1; index < length; ++index) {
            if (offset + index >= pending.size()) {
                pending.erase(0, offset);
                return decoded;
            }
            const auto byte      = static_cast<unsigned char>(pending[offset + index]);
            unsigned int minimum = 0x80U;
            unsigned int maximum = 0xbfU;
            if (index == 1) {
                if (lead == 0xe0U) {
                    minimum = 0xa0U;
                } else if (lead == 0xedU) {
                    maximum = 0x9fU;
                } else if (lead == 0xf0U) {
                    minimum = 0x90U;
                } else if (lead == 0xf4U) {
                    maximum = 0x8fU;
                }
            }
            if (byte < minimum || byte > maximum) {
                // Replace one maximal subpart. The first byte that cannot continue this sequence
                // is deliberately left for the next iteration, so valid following text is kept.
                decoded.append(kUtf8Replacement);
                offset += index;
                malformed = true;
                break;
            }
        }
        if (malformed) { continue; }

        decoded.append(pending, offset, length);
        offset += length;
    }
    pending.clear();
    return decoded;
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker,
                                  bool allow_complete = false) {
    const std::size_t maximum = std::min(text.size(), marker.size());
    for (std::size_t size = maximum; size != 0; --size) {
        if (!allow_complete && size == marker.size()) { continue; }
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

constexpr bool is_close_separator(char byte) noexcept {
    return byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n';
}

bool is_format_whitespace_only(std::string_view text) noexcept {
    for (const char byte : text) {
        if (byte != ' ' && byte != '\t' && byte != '\r' && byte != '\n') { return false; }
    }
    return true;
}

struct ReasoningCloseScan {
    std::size_t close = std::string::npos;
    std::size_t hold  = 0;
};

// Generated prose can quote the close marker while discussing the protocol, so a marker counts as
// the reasoning close only when it is followed by format whitespace, by the tool-call opener, or
// by the implicit end of the turn. A marker inside quoted text is followed by punctuation or an
// escaped newline and stays in the reasoning channel. Markers at the end of the available bytes
// stay pending until their following byte arrives, which keeps a quoted marker from closing the
// channel across a token round. The tool opener counts as a boundary because the grammar places
// a tool call immediately after the close marker, so models routinely emit
// "</think><tool_call>" without intervening whitespace; holding for whitespace that never comes
// would swallow the entire call into the reasoning channel, where the tool decoder never sees
// it. A marker followed by a partially buffered opener stays pending until the opener resolves.
ReasoningCloseScan scan_reasoning_close(std::string_view text, bool implicit_end) {
    std::size_t search = 0;
    for (;;) {
        const std::size_t marker = text.find(kThinkClose, search);
        if (marker == std::string_view::npos) { break; }
        const std::size_t after = marker + kThinkClose.size();
        if (after == text.size()) {
            if (implicit_end) { return ReasoningCloseScan{.close = marker}; }
            return ReasoningCloseScan{.hold = text.size() - marker};
        }
        if (is_close_separator(text[after])) { return ReasoningCloseScan{.close = marker}; }
        const std::string_view rest = text.substr(after);
        if (rest == kToolOpen) { return ReasoningCloseScan{.close = marker}; }
        if (rest.size() < kToolOpen.size() && kToolOpen.starts_with(rest)) {
            return ReasoningCloseScan{.hold = text.size() - marker};
        }
        search = after;
    }
    return ReasoningCloseScan{.hold = longest_suffix_prefix(text, kThinkClose, true)};
}

template <std::size_t Size>
consteval std::array<std::size_t, Size> make_prefix_failure_table(std::string_view pattern) {
    std::array<std::size_t, Size> failure{};
    for (std::size_t index = 1; index < Size; ++index) {
        std::size_t matched = failure[index - 1U];
        while (matched != 0 && pattern[index] != pattern[matched]) {
            matched = failure[matched - 1U];
        }
        if (pattern[index] == pattern[matched]) { ++matched; }
        failure[index] = matched;
    }
    return failure;
}

struct PrefixExecutionTracker {
    static constexpr std::string_view kBoundary = fi::kCanonicalReasoningCloseSerialization;
    static constexpr auto kFailure = make_prefix_failure_table<kBoundary.size()>(kBoundary);

    // Returns the byte offset immediately after the first completed boundary in this token.
    [[nodiscard]] std::optional<std::size_t> feed(std::string_view bytes) noexcept {
        if (!tracking) { return std::nullopt; }
        for (std::size_t offset = 0; offset < bytes.size(); ++offset) {
            const char byte = bytes[offset];
            while (matched != 0 && byte != kBoundary[matched]) { matched = kFailure[matched - 1U]; }
            if (byte == kBoundary[matched]) { ++matched; }
            if (matched != kBoundary.size()) { continue; }
            tracking = false;
            matched  = 0;
            return offset + 1U;
        }
        return std::nullopt;
    }

    std::size_t matched = 0;
    bool tracking       = false;
};

struct DecoderState {
    bool in_reasoning                  = false;
    std::string utf8_pending;
    std::string think_marker_pending;
    std::array<std::string, 2> stop_pending{};
    bool strip_content_leading     = false;
    bool terminal                  = false;
    std::uint64_t decoded_bytes    = 0;
    std::uint32_t reasoning_tokens = 0;
    std::optional<std::uint32_t> matched_stop_order;
    // While set, an engine-injected control span publishes to this channel without re-opening
    // the reasoning state; the decoder state itself is left for the surrounding model output.
    std::optional<OutputChannel> forced_channel;
};

struct SemanticThinkingState {
    std::optional<std::uint32_t> budget;
    std::string close_pending;
    std::uint32_t model_thinking_tokens = 0;
    std::uint32_t injected_tokens       = 0;
    bool in_reasoning                   = false;
    bool control_pending                = false;
    bool retry_pending                  = false;
    bool retry_attempted                = false;
    bool applied                        = false;
};

void feed_semantic_thinking(SemanticThinkingState& state, std::string_view bytes) {
    if (!state.in_reasoning || bytes.empty()) { return; }
    state.close_pending.append(bytes);
    const ReasoningCloseScan scan = scan_reasoning_close(state.close_pending, false);
    if (scan.close != std::string::npos) {
        state.close_pending.clear();
        state.in_reasoning    = false;
        state.control_pending = false;
        return;
    }
    state.close_pending.erase(0, state.close_pending.size() - scan.hold);
}

struct StopMatch {
    bool found                      = false;
    std::uint32_t committed_tokens  = 0;
    std::uint64_t byte_cut          = 0;
    std::uint32_t declaration_order = 0;
    PublishedOutput output;
};

bool stop_match_precedes(std::uint32_t committed_tokens, std::uint64_t byte_cut,
                         std::uint32_t declaration_order, const StopMatch& current) noexcept {
    if (!current.found) { return true; }
    if (committed_tokens != current.committed_tokens) {
        return committed_tokens < current.committed_tokens;
    }
    if (byte_cut != current.byte_cut) { return byte_cut < current.byte_cut; }
    return declaration_order < current.declaration_order;
}

std::size_t stop_hold_size(std::string_view text, OutputChannel channel, const StopPolicy& policy) {
    std::size_t hold = 0;
    for (const StopString& stop : policy.strings) {
        if (stop.channel != channel) { continue; }
        hold = std::max(hold, longest_suffix_prefix(text, stop.text));
    }
    return hold;
}

void feed_channel(DecoderState& state, OutputChannel channel, std::string_view text,
                  const StopPolicy& policy, PublishedOutput& emitted,
                  std::uint32_t committed_tokens, StopMatch* best_match) {
    if (text.empty()) { return; }
    std::string combined          = state.stop_pending[channel_index(channel)];
    const std::size_t old_pending = combined.size();
    combined.append(text);
    const std::uint64_t combined_start = state.decoded_bytes - old_pending;

    if (best_match != nullptr) {
        for (std::size_t declaration = 0; declaration < policy.strings.size(); ++declaration) {
            const StopString& stop = policy.strings[declaration];
            if (stop.channel != channel) { continue; }
            const std::size_t found = combined.find(stop.text);
            if (found == std::string::npos) { continue; }
            const std::uint64_t byte_cut = combined_start + found;
            const auto order             = static_cast<std::uint32_t>(declaration);
            if (!stop_match_precedes(committed_tokens, byte_cut, order, *best_match)) { continue; }

            PublishedOutput candidate = emitted;
            append_delta(candidate, channel, combined.substr(0, found));
            if (stop.include_in_output) { append_delta(candidate, channel, stop.text); }
            *best_match = StopMatch{.found             = true,
                                    .committed_tokens  = committed_tokens,
                                    .byte_cut          = byte_cut,
                                    .declaration_order = order,
                                    .output            = std::move(candidate)};
        }
    }

    const std::size_t hold = stop_hold_size(combined, channel, policy);
    append_delta(emitted, channel, combined.substr(0, combined.size() - hold));
    state.stop_pending[channel_index(channel)] = combined.substr(combined.size() - hold);
    state.decoded_bytes += text.size();
}

void close_channel(DecoderState& state, OutputChannel channel, PublishedOutput& emitted) {
    std::string& pending = state.stop_pending[channel_index(channel)];
    append_delta(emitted, channel, std::move(pending));
    pending.clear();
}

void feed_content(DecoderState& state, std::string text, const StopPolicy& policy,
                  PublishedOutput& emitted, std::uint32_t committed_tokens, StopMatch* best_match) {
    if (state.strip_content_leading) {
        std::size_t begin = 0;
        while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
            ++begin;
        }
        text.erase(0, begin);
        if (!text.empty()) { state.strip_content_leading = false; }
    }
    feed_channel(state, OutputChannel::Content, text, policy, emitted, committed_tokens,
                 best_match);
}

void feed_decoded_text(DecoderState& state, std::string_view text, const StopPolicy& policy,
                       PublishedOutput& emitted, std::uint32_t committed_tokens,
                       StopMatch* best_match) {
    if (state.forced_channel.has_value()) {
        feed_channel(state, *state.forced_channel, text, policy, emitted, committed_tokens,
                     best_match);
        return;
    }
    if (!state.in_reasoning) {
        feed_content(state, std::string(text), policy, emitted, committed_tokens, best_match);
        return;
    }

    state.think_marker_pending.append(text);
    const ReasoningCloseScan scan = scan_reasoning_close(state.think_marker_pending, false);
    if (scan.close != std::string::npos) {
        feed_channel(state, OutputChannel::Reasoning,
                     std::string_view(state.think_marker_pending).substr(0, scan.close), policy,
                     emitted, committed_tokens, best_match);
        close_channel(state, OutputChannel::Reasoning, emitted);
        std::string content = state.think_marker_pending.substr(scan.close + kThinkClose.size());
        state.think_marker_pending.clear();
        state.in_reasoning          = false;
        state.strip_content_leading = true;
        feed_content(state, std::move(content), policy, emitted, committed_tokens, best_match);
        return;
    }

    const std::size_t safe = state.think_marker_pending.size() - scan.hold;
    feed_channel(state, OutputChannel::Reasoning,
                 std::string_view(state.think_marker_pending).substr(0, safe), policy, emitted,
                 committed_tokens, best_match);
    state.think_marker_pending.erase(0, safe);
}

void feed_token_bytes(DecoderState& state, std::string_view bytes, const StopPolicy& policy,
                      PublishedOutput& emitted, std::uint32_t committed_tokens,
                      StopMatch* best_match) {
    state.utf8_pending.append(bytes);
    const std::string text = consume_generated_utf8(state.utf8_pending);
    feed_decoded_text(state, text, policy, emitted, committed_tokens, best_match);
}

void terminalize(DecoderState& state, const StopPolicy& policy, PublishedOutput& emitted,
                 std::uint32_t committed_tokens) {
    if (!state.utf8_pending.empty()) {
        // A token budget can end between byte-level tokens of one code point.
        // Publish the standard replacement character rather than an invalid
        // UTF-8 suffix; the logical token prefix remains exact.
        state.utf8_pending.clear();
        feed_decoded_text(state, kUtf8Replacement, policy, emitted, committed_tokens, nullptr);
    }
    if (state.in_reasoning) {
        // A close marker still pending at the end of the turn is the model's implicit close.
        const ReasoningCloseScan scan = scan_reasoning_close(state.think_marker_pending, true);
        const std::size_t split =
            scan.close != std::string::npos ? scan.close : state.think_marker_pending.size();
        feed_channel(state, OutputChannel::Reasoning,
                     std::string_view(state.think_marker_pending).substr(0, split), policy, emitted,
                     committed_tokens, nullptr);
        close_channel(state, OutputChannel::Reasoning, emitted);
        if (scan.close != std::string::npos) {
            std::string content =
                state.think_marker_pending.substr(scan.close + kThinkClose.size());
            state.think_marker_pending.clear();
            state.in_reasoning          = false;
            state.strip_content_leading = true;
            feed_content(state, std::move(content), policy, emitted, committed_tokens, nullptr);
        } else {
            state.think_marker_pending.clear();
        }
    }
    close_channel(state, OutputChannel::Content, emitted);
    state.stop_pending = {};
    state.terminal     = true;
}

DecoderState terminal_state(DecoderState state) {
    state.utf8_pending.clear();
    state.think_marker_pending.clear();
    state.stop_pending = {};
    state.terminal     = true;
    return state;
}

} // namespace

class OutputSession::Impl {
public:
    Impl(std::shared_ptr<const fi::Tokenizer> tokenizer_, StopPolicy policy_, OutputOptions output,
         bool starts_in_reasoning, ThinkingControlOptions thinking,
         std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens_,
         std::shared_ptr<const std::vector<TokenId>> retry_control_tokens_,
         std::shared_ptr<const fi::ToolCallOutputContract> tool_call_output_)
        : tokenizer(std::move(tokenizer_)), policy(std::move(policy_)),
          thinking_control_tokens(std::move(thinking_control_tokens_)),
          retry_control_tokens(std::move(retry_control_tokens_)),
          preserve_special(output.raw || output.preserve_special_tokens),
          split_reasoning(starts_in_reasoning && !output.raw),
          tool_call_contract(output.raw ? nullptr : tool_call_output_),
          tool_name_max_length(output.tool_name_max_length),
          tool_call_output(output.raw ? nullptr : std::move(tool_call_output_),
                           output.tool_name_max_length) {
        if (thinking.budget && *thinking.budget == 0) {
            throw std::invalid_argument("thinking budget must be positive");
        }
        state.in_reasoning        = split_reasoning;
        prefix_execution.tracking = starts_in_reasoning;
        semantic.budget           = thinking.budget;
        // The semantic tracker counts model-origin thinking tokens whenever the response starts
        // in thinking, capped or not: the count is reported per request, and it must not depend
        // on the presentation decoder, which raw output disables. A budget additionally caps it.
        // Its per-token work is one bounded scan for the close marker.
        semantic.in_reasoning = starts_in_reasoning;
    }

    // A thinking model can narrate a complete call inside its reasoning channel and close
    // thinking without emitting it in content. The presentation decoder only parses the content
    // channel, so such a turn ends with no structured call even though the model intended one.
    // When the content channel is empty or whitespace-only and the reasoning channel ends with a
    // well-formed call suffix, promote that suffix to a structured call. parse_qwen_tool_call_output
    // already refuses a call that is followed by further text, so only a terminal, well-formed
    // region is ever salvaged.
    void salvage_reasoning_tool_call() {
        if (!tool_calls.empty() || tool_call_contract == nullptr || reasoning_text.empty() ||
            !is_format_whitespace_only(content_text)) {
            return;
        }
        const fi::ParsedToolCallOutput parsed =
            fi::parse_qwen_tool_call_output(reasoning_text, tool_name_max_length,
                                            *tool_call_contract);
        if (!parsed.is_tool_call_response || parsed.tool_calls.empty()) { return; }
        tool_calls                           = std::move(parsed.tool_calls);
        tool_call_parse                      = parsed.diagnostics;
        tool_call_parse.salvaged_from_reasoning = true;
    }

    // A turn that ends on its own stop token without a message - whitespace-only content, no
    // tool call, and no reasoning-narrated call to salvage - is a silent response. The
    // configured message is re-injected once, as a forced control span, only when the thinking
    // budget is set and the model has actually exhausted it: that is the loop signature
    // (the model spent its licensed thinking and still produced nothing). A silent stop with
    // headroom left, or without any budget, ends the turn normally, so mid-tool-loop stops are
    // never hijacked. One attempt per request.
    bool arm_empty_turn_retry(std::uint32_t remaining_after_round) {
        if (retry_control_tokens == nullptr || preview_semantic.retry_attempted) { return false; }
        if (!preview_semantic.budget ||
            preview_semantic.model_thinking_tokens < *preview_semantic.budget) {
            return false;
        }
        const std::shared_ptr<const std::vector<TokenId>>& span = preview_semantic.in_reasoning
                                                                      ? thinking_control_tokens
                                                                      : retry_control_tokens;
        if (span == nullptr || span->size() > remaining_after_round) { return false; }
        std::string all_content = content_text;
        for (const OutputDelta& delta : preview_output) {
            if (delta.channel == OutputChannel::Content) { all_content += delta.text; }
        }
        if (!is_format_whitespace_only(all_content) || !tool_calls.empty() ||
            reasoning_salvage_would_apply()) {
            return false;
        }
        preview_semantic.retry_pending = true;
        return true;
    }

    // The terminal reasoning-tool-call salvage would find a structured call on the
    // committed-plus-preview reasoning text. Such a turn ends with a message via that salvage,
    // so it must not be retried.
    bool reasoning_salvage_would_apply() const {
        if (tool_call_contract == nullptr) { return false; }
        std::string all_reasoning = reasoning_text;
        for (const OutputDelta& delta : preview_output) {
            if (delta.channel == OutputChannel::Reasoning) { all_reasoning += delta.text; }
        }
        if (all_reasoning.empty()) { return false; }
        const fi::ParsedToolCallOutput parsed =
            fi::parse_qwen_tool_call_output(all_reasoning, tool_name_max_length,
                                            *tool_call_contract);
        return parsed.is_tool_call_response && !parsed.tool_calls.empty();
    }

    std::shared_ptr<const fi::Tokenizer> tokenizer;
    StopPolicy policy;
    std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens;
    std::shared_ptr<const std::vector<TokenId>> retry_control_tokens;
    bool preserve_special = false;
    bool split_reasoning  = false;
    DecoderState state;
    DecoderState preview_state;
    SemanticThinkingState semantic;
    SemanticThinkingState preview_semantic;
    PrefixExecutionTracker prefix_execution;
    PrefixExecutionTracker preview_prefix_execution;
    std::optional<std::uint32_t> preview_execution_split_after;
    PublishedOutput preview_output;
    std::shared_ptr<const fi::ToolCallOutputContract> tool_call_contract;
    std::size_t tool_name_max_length = 0;
    fi::ToolCallOutputDecoder tool_call_output;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    std::string reasoning_text;
    std::string content_text;
    bool preview_ready = false;
};

PublishedOutput::PublishedOutput(PublishedOutput&& other) noexcept
    : values_(std::move(other.values_)), size_(std::exchange(other.size_, 0)) {}

PublishedOutput& PublishedOutput::operator=(PublishedOutput&& other) noexcept {
    if (this != &other) {
        values_ = std::move(other.values_);
        size_   = std::exchange(other.size_, 0);
    }
    return *this;
}

void PublishedOutput::clear() noexcept {
    for (std::size_t index = 0; index < size_; ++index) { values_[index] = {}; }
    size_ = 0;
}

void PublishedOutput::push_back(OutputDelta value) {
    if (size_ == values_.size()) {
        throw std::logic_error("output decoder produced more than two channel transitions");
    }
    values_[size_++] = std::move(value);
}

OutputSession::OutputSession() noexcept                           = default;
OutputSession::~OutputSession()                                   = default;
OutputSession::OutputSession(OutputSession&&) noexcept            = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept = default;

OutputSession::OutputSession(
    std::shared_ptr<const frontend::Tokenizer> tokenizer, StopPolicy policy, OutputOptions output,
    bool starts_in_reasoning, ThinkingControlOptions thinking,
    std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens,
    std::shared_ptr<const std::vector<TokenId>> retry_control_tokens,
    std::shared_ptr<const frontend::ToolCallOutputContract> tool_call_output)
    : impl_(std::make_unique<Impl>(
          std::move(tokenizer), std::move(policy), output, starts_in_reasoning, thinking,
          std::move(thinking_control_tokens), std::move(retry_control_tokens),
          std::move(tool_call_output))) {}

runtime::OutputDecision OutputSession::preview_model(std::span<const TokenId> tokens,
                                                     std::uint32_t total_budget_remaining,
                                                     FinishReason limit_reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (impl_->semantic.control_pending || impl_->semantic.retry_pending) {
        throw std::logic_error("model output cannot advance while thinking control is pending");
    }
    if (tokens.empty()) {
        throw std::invalid_argument("cannot preview an empty generated-token round");
    }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("generated-token round exceeds the remaining budget");
    }
    if (limit_reason != FinishReason::OutputLimit &&
        limit_reason != FinishReason::ContextCapacity) {
        throw std::invalid_argument("generated-token budget has an invalid limit reason");
    }

    impl_->preview_state            = impl_->state;
    impl_->preview_semantic         = impl_->semantic;
    impl_->preview_prefix_execution = impl_->prefix_execution;
    impl_->preview_execution_split_after.reset();
    impl_->preview_output.clear();

    const auto complete = [&](std::uint32_t count, FinishReason reason,
                              runtime::ContinuationAction continuation =
                                  runtime::ContinuationAction::Decode) {
        if (reason != FinishReason::None) { impl_->preview_semantic.control_pending = false; }
        if (impl_->preview_execution_split_after && *impl_->preview_execution_split_after > count) {
            throw std::logic_error("prefix execution split exceeds the accepted token prefix");
        }
        impl_->preview_ready = true;
        return runtime::OutputDecision{
            .accepted_tokens              = count,
            .finish_reason                = reason,
            .continuation                 = continuation,
            .prefix_execution_split_after = impl_->preview_execution_split_after,
        };
    };

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::uint32_t count          = static_cast<std::uint32_t>(index + 1);
        const TokenId token                = tokens[index];
        const fi::DecodedTokenView decoded = impl_->tokenizer->decoded_token(token);

        if (const auto boundary = impl_->preview_prefix_execution.feed(decoded.bytes);
            boundary && *boundary == decoded.bytes.size()) {
            impl_->preview_execution_split_after = count;
        }

        if (impl_->preview_state.in_reasoning) { ++impl_->preview_state.reasoning_tokens; }
        if (impl_->preview_semantic.in_reasoning) {
            ++impl_->preview_semantic.model_thinking_tokens;
            if (impl_->preview_semantic.budget &&
                impl_->preview_semantic.model_thinking_tokens > *impl_->preview_semantic.budget) {
                throw std::logic_error("model output exceeded the licensed thinking budget");
            }
            feed_semantic_thinking(impl_->preview_semantic, decoded.bytes);
        }

        const bool stop_token =
            std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(), token) !=
            impl_->policy.token_ids.end();
        DecoderState before_state;
        PublishedOutput before_output;
        if (stop_token && !impl_->policy.publish_stop_token) {
            before_state  = impl_->preview_state;
            before_output = impl_->preview_output;
        }

        StopMatch match;
        const std::string_view bytes =
            !impl_->preserve_special && decoded.special ? std::string_view{} : decoded.bytes;
        feed_token_bytes(impl_->preview_state, bytes, impl_->policy, impl_->preview_output, count,
                         &match);

        if (match.found) {
            impl_->preview_state = terminal_state(std::move(impl_->preview_state));
            impl_->preview_state.matched_stop_order = match.declaration_order;
            impl_->preview_output                   = std::move(match.output);
            return complete(match.committed_tokens, FinishReason::StopString);
        }

        if (stop_token) {
            if (!impl_->policy.publish_stop_token) {
                impl_->preview_state  = std::move(before_state);
                impl_->preview_output = std::move(before_output);
            }
            if (impl_->arm_empty_turn_retry(total_budget_remaining - count)) {
                return complete(count, FinishReason::None,
                                runtime::ContinuationAction::ApplyTargetControl);
            }
            terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count);
            return complete(count, FinishReason::StopToken);
        }
    }

    const auto count = static_cast<std::uint32_t>(tokens.size());
    if (tokens.size() == total_budget_remaining) {
        terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count);
        return complete(count, limit_reason);
    }
    if (impl_->preview_semantic.in_reasoning && impl_->preview_semantic.budget &&
        impl_->preview_semantic.model_thinking_tokens == *impl_->preview_semantic.budget) {
        impl_->preview_semantic.control_pending = true;
        return complete(count, FinishReason::None, runtime::ContinuationAction::ApplyTargetControl);
    }
    return complete(count, FinishReason::None);
}

std::uint32_t
OutputSession::model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept {
    if (impl_ == nullptr || !impl_->semantic.budget || !impl_->semantic.in_reasoning ||
        impl_->semantic.applied) {
        return total_budget_remaining;
    }
    if (impl_->semantic.control_pending ||
        impl_->semantic.model_thinking_tokens >= *impl_->semantic.budget) {
        return 0;
    }
    return std::min(total_budget_remaining,
                    *impl_->semantic.budget - impl_->semantic.model_thinking_tokens);
}

std::span<const TokenId> OutputSession::pending_control_tokens() const noexcept {
    if (impl_ == nullptr) { return {}; }
    if (impl_->semantic.control_pending) {
        return impl_->thinking_control_tokens ? *impl_->thinking_control_tokens
                                              : std::span<const TokenId>{};
    }
    if (impl_->semantic.retry_pending) {
        const std::shared_ptr<const std::vector<TokenId>>& span =
            impl_->semantic.in_reasoning ? impl_->thinking_control_tokens
                                         : impl_->retry_control_tokens;
        return span ? *span : std::span<const TokenId>{};
    }
    return {};
}

std::uint32_t OutputSession::control_suffix_tokens() const noexcept {
    if (impl_ == nullptr || !impl_->thinking_control_tokens) { return 0; }
    const SemanticThinkingState& s = impl_->semantic;
    const bool cutoff_armed = s.budget.has_value() && (s.control_pending || s.in_reasoning);
    // The retry can only arm once the thinking budget is exhausted, so its reservation only has
    // to persist from exhaustion until the one retry is spent (any other path ends the request).
    const bool retry_armed = impl_->retry_control_tokens != nullptr && s.budget.has_value() &&
                             s.model_thinking_tokens >= *s.budget && !s.retry_attempted;
    return (cutoff_armed || retry_armed)
               ? static_cast<std::uint32_t>(impl_->thinking_control_tokens->size())
               : 0U;
}

runtime::OutputDecision OutputSession::preview_control(std::span<const TokenId> tokens,
                                                       std::uint32_t total_budget_remaining) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    const std::span<const TokenId> expected = pending_control_tokens();
    if (expected.empty() || tokens.size() != expected.size() ||
        !std::equal(tokens.begin(), tokens.end(), expected.begin())) {
        throw std::invalid_argument("thinking control preview requires the exact pending span");
    }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("thinking control span exceeds the remaining output budget");
    }

    impl_->preview_state            = impl_->state;
    impl_->preview_semantic         = impl_->semantic;
    impl_->preview_prefix_execution = impl_->prefix_execution;
    impl_->preview_execution_split_after.reset();
    impl_->preview_output.clear();
    const bool empty_turn_retry = impl_->preview_semantic.retry_pending;
    // A retry span following a naturally closed (or never opened) thinking phase has no close
    // marker of its own; publish it in the reasoning channel without re-opening the state.
    const bool forced_reasoning = empty_turn_retry && !impl_->preview_state.in_reasoning;
    if (forced_reasoning) { impl_->preview_state.forced_channel = OutputChannel::Reasoning; }
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const TokenId token                = tokens[index];
        const fi::DecodedTokenView decoded = impl_->tokenizer->decoded_token(token);
        if (const auto boundary = impl_->preview_prefix_execution.feed(decoded.bytes);
            boundary && *boundary == decoded.bytes.size()) {
            impl_->preview_execution_split_after = static_cast<std::uint32_t>(index + 1U);
        }
        if (impl_->preview_state.in_reasoning || forced_reasoning) {
            ++impl_->preview_state.reasoning_tokens;
        }
        feed_semantic_thinking(impl_->preview_semantic, decoded.bytes);
        const std::string_view presentation_bytes =
            !impl_->preserve_special && decoded.special ? std::string_view{} : decoded.bytes;
        feed_token_bytes(impl_->preview_state, presentation_bytes, impl_->policy,
                         impl_->preview_output, static_cast<std::uint32_t>(index + 1), nullptr);
    }
    if (forced_reasoning) {
        impl_->preview_state.forced_channel = std::nullopt;
        if (!impl_->preview_state.utf8_pending.empty()) {
            throw std::logic_error("injected retry span did not end on a UTF-8 boundary");
        }
    }
    if (impl_->preview_semantic.in_reasoning) {
        throw std::logic_error("canonical thinking control did not close the thinking phase");
    }
    if (impl_->split_reasoning && impl_->preview_state.in_reasoning) {
        throw std::logic_error("canonical thinking control did not close the reasoning channel");
    }
    impl_->preview_semantic.control_pending = false;
    impl_->preview_semantic.retry_pending   = false;
    if (empty_turn_retry) {
        impl_->preview_semantic.retry_attempted = true;
    } else {
        impl_->preview_semantic.applied = true;
    }
    impl_->preview_semantic.injected_tokens += static_cast<std::uint32_t>(tokens.size());
    impl_->preview_ready                    = true;
    return runtime::OutputDecision{
        .accepted_tokens              = static_cast<std::uint32_t>(tokens.size()),
        .prefix_execution_split_after = impl_->preview_execution_split_after,
    };
}

void OutputSession::validate_generation_capacity(std::uint32_t effective_output_tokens) const {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (!impl_->semantic.budget || !impl_->semantic.in_reasoning ||
        effective_output_tokens <= *impl_->semantic.budget) {
        return;
    }
    const std::uint64_t remaining =
        static_cast<std::uint64_t>(effective_output_tokens) - *impl_->semantic.budget;
    const std::uint64_t required =
        static_cast<std::uint64_t>(impl_->thinking_control_tokens->size()) + 1U;
    if (remaining < required) {
        throw std::invalid_argument(
            "effective output capacity after the thinking budget must fit the complete control "
            "suffix and one post-close model token");
    }
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (reason == FinishReason::None || reason == FinishReason::StopString ||
        reason == FinishReason::StopToken) {
        throw std::invalid_argument("invalid between-round terminal decoder reason");
    }
    impl_->preview_state            = impl_->state;
    impl_->preview_semantic         = impl_->semantic;
    impl_->preview_prefix_execution = impl_->prefix_execution;
    impl_->preview_execution_split_after.reset();
    impl_->preview_semantic.control_pending = false;
    impl_->preview_semantic.retry_pending   = false;
    impl_->preview_output.clear();
    terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, 0);
    impl_->preview_ready = true;
    return runtime::OutputDecision{.accepted_tokens = 0, .finish_reason = reason};
}

PublishedOutput OutputSession::commit_preview() {
    if (impl_ == nullptr || !impl_->preview_ready) { std::terminate(); }
    using std::swap;
    swap(impl_->state, impl_->preview_state);
    swap(impl_->semantic, impl_->preview_semantic);
    swap(impl_->prefix_execution, impl_->preview_prefix_execution);
    PublishedOutput output = std::move(impl_->preview_output);
    impl_->preview_output.clear();
    impl_->preview_ready = false;

    for (OutputDelta& delta : output) {
        if (delta.channel == OutputChannel::Content) {
            delta.text = impl_->tool_call_output.feed(delta.text);
        }
    }
    if (impl_->state.terminal) {
        fi::ToolCallOutputDecoder::Terminal terminal = impl_->tool_call_output.finish();
        impl_->tool_calls                            = std::move(terminal.tool_calls);
        impl_->tool_call_parse                       = terminal.diagnostics;
        if (!terminal.content.empty()) {
            OutputDelta* content = nullptr;
            for (OutputDelta& delta : output) {
                if (delta.channel == OutputChannel::Content) { content = &delta; }
            }
            if (content != nullptr) {
                content->text += terminal.content;
            } else {
                output.push_back(OutputDelta{.channel = OutputChannel::Content,
                                             .text    = std::move(terminal.content)});
            }
        }
    }

    // Accumulate the full published text of each channel so the terminal salvage can consider
    // the whole turn, not just the final commit.
    for (const OutputDelta& delta : output) {
        if (delta.channel == OutputChannel::Reasoning) {
            impl_->reasoning_text += delta.text;
        } else {
            impl_->content_text += delta.text;
        }
    }
    if (impl_->state.terminal) {
        impl_->salvage_reasoning_tool_call();
    }
    return output;
}

std::vector<GeneratedToolCall> OutputSession::take_tool_calls() noexcept {
    return impl_ != nullptr ? std::move(impl_->tool_calls) : std::vector<GeneratedToolCall>{};
}

ToolCallParseDiagnostics OutputSession::tool_call_parse_diagnostics() const noexcept {
    return impl_ != nullptr ? impl_->tool_call_parse : ToolCallParseDiagnostics{};
}

std::uint32_t OutputSession::reasoning_tokens() const noexcept {
    return impl_ != nullptr ? impl_->state.reasoning_tokens : 0;
}

ThinkingBudgetStats OutputSession::thinking_stats() const noexcept {
    if (impl_ == nullptr) { return {}; }
    return ThinkingBudgetStats{
        .configured_budget     = impl_->semantic.budget,
        .model_thinking_tokens = impl_->semantic.model_thinking_tokens,
        .injected_tokens       = impl_->semantic.injected_tokens,
        .applied               = impl_->semantic.applied,
        .empty_turn_retry      = impl_->semantic.retry_attempted,
    };
}

std::optional<std::string> OutputSession::matched_stop_string() const {
    if (impl_ == nullptr || !impl_->state.matched_stop_order) { return std::nullopt; }
    const std::size_t index = *impl_->state.matched_stop_order;
    if (index >= impl_->policy.strings.size()) {
        throw std::logic_error("matched stop declaration is outside the stop policy");
    }
    return impl_->policy.strings[index].text;
}

} // namespace ninfer::models::qwen3_5
