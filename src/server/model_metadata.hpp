#pragma once

#include "AutoModel/reasoning_efforts.hpp"
#include "nlohmann/json.hpp"
#include "utils/debug_utils.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>

namespace flm {

inline bool has_catalog_label(const nlohmann::json& entry, const char* label) {
    const auto labels = entry.find("label");
    return labels != entry.end() && labels->is_array() &&
           std::find(labels->begin(), labels->end(), label) != labels->end();
}

inline nlohmann::json read_model_metadata(const std::filesystem::path& path) {
    std::error_code error;
    const auto status = std::filesystem::status(path, error);
    if (status.type() == std::filesystem::file_type::not_found) return {};
    constexpr std::size_t max_bytes = 16 * 1024;
    std::array<char, max_bytes + 1> bytes;
    if (error || !std::filesystem::is_regular_file(status)) {
        header_print_r("WARNING", "Cannot read model metadata: " << path);
        return {};
    }
    std::ifstream input(path, std::ios::binary);
    input.read(bytes.data(), bytes.size());
    if (input.bad() || !input.eof() || input.gcount() > max_bytes) {
        header_print_r("WARNING", "Unreadable or oversized model metadata: " << path);
        return {};
    }
    auto metadata = nlohmann::json::parse(bytes.data(), bytes.data() + input.gcount(), nullptr, false);
    if (!metadata.is_object()) {
        header_print_r("WARNING", "Expected a JSON object in model metadata: " << path);
        return {};
    }
    return metadata;
}

inline bool supports_num_ctx(const nlohmann::json& entry) {
    const auto family = entry.value("details", nlohmann::json::object()).value("family", std::string());
    if (family.empty() || family == "whisper-v3" || family == "r2t2" ||
        family == "embed-gemma" || family == "d3") return false;
    return !has_catalog_label(entry, "single-turn");
}

inline std::optional<int> model_position_limit(const std::filesystem::path& load_package) {
    const auto config = read_model_metadata(load_package / "config.json");
    const auto position_limit = [](const nlohmann::json& section) -> std::optional<int> {
        const auto maximum = section.find("max_position_embeddings");
        if (maximum == section.end() || !maximum->is_number_integer() || *maximum <= 0 ||
            (maximum->is_number_unsigned() && maximum->get<std::uint64_t>() > std::numeric_limits<int>::max()) ||
            (!maximum->is_number_unsigned() && maximum->get<std::int64_t>() > std::numeric_limits<int>::max()))
            return std::nullopt;
        return maximum->get<int>();
    };
    const auto text = config.find("text_config");
    if (text != config.end() && text->is_object()) {
        if (auto limit = position_limit(*text)) return limit;
    }
    return position_limit(config);
}

inline nlohmann::json model_capabilities(const nlohmann::json& entry,
                                         const std::filesystem::path& package,
                                         const std::filesystem::path& load_package) {
    auto result = nlohmann::json::object();
    if (supports_num_ctx(entry)) result["context_window_override"] = true;
    if (has_catalog_label(entry, "tool-calling")) result["tools"] = true;
    if (const auto labels = entry.find("label"); labels != entry.end() && labels->is_array()) {
        result["audio"] = has_catalog_label(entry, "audio");
    }
    const auto vlm = entry.find("vlm");
    if (vlm != entry.end() && vlm->is_boolean()) result["vlm"] = *vlm;
    const auto context = entry.find("default_context_length");
    if (context != entry.end() && context->is_number_integer() && *context > 0) {
        result["default_context_length"] = *context;
    }
    if (const auto details = entry.find("details"); details != entry.end() && details->is_object()) {
        const auto think = details->find("think_toggleable");
        if (think != details->end() && think->is_boolean()) result["think"] = *think;
        if (const auto family = details->find("family"); family != details->end()) {
            if (*family == "gpt-oss") result["reasoning_effort"] = gpt_oss_efforts;
            else if (*family == "hy-memory") result["reasoning_effort"] = hy_memory_efforts;
        }
    }
    if (auto limit = model_position_limit(load_package)) result["max_position_embeddings"] = *limit;

    const auto path = package / "model.json";
    const auto metadata = read_model_metadata(path);
    for (const char* key : {"think", "tools"}) {
        if (result.contains(key)) continue;
        const auto value = metadata.find(key);
        if (value == metadata.end()) continue;
        if (value->is_boolean()) result[key] = *value;
        else header_print_r("WARNING", "Expected boolean " << key << " in model metadata: " << path);
    }
    return result;
}

}  // namespace flm
