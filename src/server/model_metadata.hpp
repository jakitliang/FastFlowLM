#pragma once

// FLM catalog facts, installed model configuration, and request window validation.
// This module deliberately has no accelerator dependency: metadata must not load a model.
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace flm::metadata {
using json = nlohmann::ordered_json;

inline std::optional<int> positive_integer(const json& value) {
    if (!value.is_number_integer()) return std::nullopt;
    if (value.is_number_unsigned()) {
        const auto n = value.get<std::uint64_t>();
        if (n > 0 && n <= std::numeric_limits<int>::max()) return static_cast<int>(n);
    } else {
        const auto n = value.get<std::int64_t>();
        if (n > 0 && n <= std::numeric_limits<int>::max()) return static_cast<int>(n);
    }
    return std::nullopt;
}

inline bool has_label(const json& catalog, const std::string& label) {
    const auto labels = catalog.find("label");
    if (labels == catalog.end() || !labels->is_array()) return false;
    for (const auto& value : *labels) if (value == label) return true;
    return false;
}

inline json describe(const json& catalog, const json& config, int startup_context,
                     std::uint32_t backend_limit = 0) {
    json result = json::object();
    if (catalog.contains("details") && catalog["details"].is_object()) result["details"] = catalog["details"];
    if (catalog.contains("size")) result["size"] = catalog["size"];
    if (catalog.contains("modified_at")) result["modified_at"] = catalog["modified_at"];
    const bool embedding = has_label(catalog, "embeddings");
    const bool transcription = catalog.value("asr", false);
    const bool decision = has_label(catalog, "decision");
    const bool causal_chat = !embedding && !transcription && !decision;
    result["type"] = embedding ? "embedding" : transcription ? "audio" : decision ? "decision" : "chat";
    json capabilities = json::array({embedding ? "embedding" : transcription ? "transcription"
        : decision ? "decision" : "completion"});
    if (catalog.value("vlm", false) || has_label(catalog, "vision")) capabilities.push_back("vision");
    if (has_label(catalog, "tool-calling")) capabilities.push_back("tools");
    const auto details = catalog.value("details", json::object());
    if (details.value("think", false)) capabilities.push_back("thinking");
    result["capabilities"] = capabilities;
    if (details.contains("think_toggleable") && details["think_toggleable"].is_boolean())
        result["thinking_toggleable"] = details["think_toggleable"];

    // A catalog default is not an architectural maximum. Only expose the latter
    // when the installed artifact's config actually declares it.
    const auto& thinker = config.contains("thinker_config") && config["thinker_config"].is_object()
        ? config["thinker_config"] : config;
    const auto& text = thinker.contains("text_config") && thinker["text_config"].is_object()
        ? thinker["text_config"] : thinker;
    const auto maximum = positive_integer(text.value("max_position_embeddings", json()));
    const auto default_window = positive_integer(catalog.value("default_context_length", json()));
    const bool fixed = has_label(catalog, "single-turn");
    const auto configured = fixed || startup_context < 0 ? default_window : std::optional<int>(startup_context);
    result["context_adjustable"] = !fixed && causal_chat;
    if (default_window) result["default_context_length"] = *default_window;
    if (configured) result["configured_context_length"] = *configured;
    if (maximum) {
        result["model_context_length"] = *maximum;
        const auto text_type = text.find("model_type");
        const auto root_type = config.find("model_type");
        const std::string architecture = text_type != text.end() && text_type->is_string()
            ? text_type->get<std::string>()
            : root_type != config.end() && root_type->is_string() ? root_type->get<std::string>() : std::string();
        if (!architecture.empty()) result["model_info"] = {
            {"general.architecture", architecture}, {architecture + ".context_length", *maximum}
        };
        if (!fixed && causal_chat) {
            const auto selectable_max = backend_limit == 0 ? *maximum
                : static_cast<int>(std::min<std::uint32_t>(*maximum, backend_limit));
            result["context_capacity"] = {{"max", selectable_max}, {"request_parameter", "num_ctx"},
                {"source", "installed_config"}};
            if (configured && *configured <= selectable_max) result["context_capacity"]["default"] = *configured;
        }
    }
    if (backend_limit > 0) result["backend_context_limit"] = backend_limit;
    return result;
}

inline json read_config(const std::string& directory) {
    try {
        std::ifstream stream(directory + "/config.json");
        if (!stream) return json::object();
        auto config = json::parse(stream);
        return config.is_object() ? config : json::object();
    } catch (const json::exception&) {
        return json::object();
    }
}

// OpenAI extension: top-level num_ctx. Native chat/generate: options.num_ctx.
// Missing/null = Auto, not the previous request's override.
inline std::optional<int> request_context(const json& request) {
    const json* selected = nullptr;
    if (request.contains("num_ctx") && !request["num_ctx"].is_null()) selected = &request["num_ctx"];
    const auto options = request.find("options");
    if (options != request.end() && !options->is_null() && !options->is_object())
        throw std::invalid_argument("options must be an object");
    if (options != request.end() && options->is_object() && options->contains("num_ctx") && !(*options)["num_ctx"].is_null()) {
        if (selected && *selected != (*options)["num_ctx"]) throw std::invalid_argument("Conflicting num_ctx values");
        selected = &(*options)["num_ctx"];
    }
    if (!selected) return std::nullopt;
    const auto window = positive_integer(*selected);
    if (!window || *window < 512) throw std::invalid_argument("num_ctx must be an integer between 512 and INT_MAX");
    return window;
}

inline int resolve_request_context(const json& request, const json& description) {
    const auto window = request_context(request);
    if (!window) return -1;
    if (!description.value("context_adjustable", false))
        throw std::invalid_argument("This model has a fixed context window");
    const auto maximum = positive_integer(description.value("model_context_length", json()));
    if (maximum && *window > *maximum)
        throw std::invalid_argument("num_ctx exceeds the installed model's declared context length");
    const auto backend = positive_integer(description.value("backend_context_limit", json()));
    if (backend && *window > *backend)
        throw std::invalid_argument("num_ctx exceeds the selected backend's context limit");
    return *window;
}
}  // namespace flm::metadata