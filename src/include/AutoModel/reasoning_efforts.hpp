#pragma once

#include <array>
#include <string_view>

namespace flm {

inline constexpr std::array<std::string_view, 3> gpt_oss_efforts = {"low", "medium", "high"};
inline constexpr std::array<std::string_view, 2> hy_memory_efforts = {"low", "high"};

template <std::size_t N>
constexpr bool accepts_reasoning_effort(const std::array<std::string_view, N>& efforts, std::string_view value) {
    for (const auto effort : efforts) {
        if (effort == value) return true;
    }
    return false;
}

}  // namespace flm
