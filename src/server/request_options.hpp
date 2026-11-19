#pragma once

#include "nlohmann/json.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace flm::http_guard {

inline nlohmann::ordered_json error_body(const std::string& message, int code,
                                        const char* param,
                                        const char* type = "invalid_request_error") {
    return {{"error", {{"message", message}, {"type", type}, {"param", param}, {"code", code}}}};
}

inline bool query_has(std::string_view target, std::string_view parameter) {
    target = target.substr(0, target.find('#'));
    const auto start = target.find('?');
    if (start == std::string_view::npos) return false;
    auto query = target.substr(start + 1);
    while (!query.empty()) {
        const auto separator = query.find('&');
        if (query.substr(0, separator) == parameter) return true;
        if (separator == std::string_view::npos) break;
        query.remove_prefix(separator + 1);
    }
    return false;
}

inline std::optional<nlohmann::ordered_json> parse_num_ctx(
    const nlohmann::ordered_json& fields, const char* param, std::optional<int>& num_ctx) {
    num_ctx.reset();
    const auto it = fields.find("num_ctx");
    if (it == fields.end()) return std::nullopt;
    constexpr int max = std::numeric_limits<int>::max();
    const bool valid = it->is_number_unsigned() ? it->get<std::uint64_t>() <= max && *it != 0
        : it->is_number_integer() && it->get<std::int64_t>() > 0 && it->get<std::int64_t>() <= max;
    if (!valid)
        return error_body(std::string("'") + param + "' must be a positive integer no greater than " +
                          std::to_string(max) + ".", 400, param);
    num_ctx = it->get<int>();
    return std::nullopt;
}

}  // namespace flm::http_guard
