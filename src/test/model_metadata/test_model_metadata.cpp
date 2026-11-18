#include "model_metadata.hpp"
#include <iostream>

using flm::metadata::json;
void check(bool value) { if (!value) throw std::runtime_error("metadata assertion failed"); }
template <typename F> void rejects(F operation) {
    bool rejected = false;
    try { operation(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected);
}
int main() {
    // Synthetic FLM fixtures; no Ollama or OVMS test data/helpers.
    const json catalog = {{"default_context_length", 8192}, {"size", 1234}, {"vlm", false},
        {"details", {{"family", "fixture"}, {"think", true}, {"think_toggleable", true}}},
        {"label", {"reasoning", "tool-calling"}}};
    const auto unknown = flm::metadata::describe(catalog, json::object(), -1);
    check(!unknown.contains("context_capacity"));
    check(!unknown.contains("model_context_length"));
    check(unknown["default_context_length"] == 8192);
    check(unknown["capabilities"] == json::array({"completion", "tools", "thinking"}));
    const json config = {{"model_type", "fixture"}, {"max_position_embeddings", 65536}};
    const auto known = flm::metadata::describe(catalog, config, 16384);
    check(known["context_capacity"]["max"] == 65536);
    check(known["context_capacity"]["default"] == 16384);
    check(known["model_info"]["fixture.context_length"] == 65536);
    const auto bounded = flm::metadata::describe(catalog, config, 16384, 32768);
    check(bounded["model_context_length"] == 65536);
    check(bounded["context_capacity"]["max"] == 32768);
    rejects([&] { flm::metadata::resolve_request_context({{"num_ctx", 32769}}, bounded); });
    const auto backend_only = flm::metadata::describe(catalog, json::object(), -1, 32768);
    check(!backend_only.contains("context_capacity"));
    rejects([&] { flm::metadata::resolve_request_context({{"num_ctx", 32769}}, backend_only); });
    check(flm::metadata::resolve_request_context(json::object(), known) == -1);
    check(flm::metadata::resolve_request_context({{"num_ctx", nullptr}}, known) == -1);
    check(flm::metadata::resolve_request_context({{"num_ctx", 32768}}, known) == 32768);
    check(flm::metadata::resolve_request_context({{"options", {{"num_ctx", 32768}}}}, known) == 32768);
    rejects([&] { flm::metadata::resolve_request_context({{"num_ctx", 65537}}, known); });
    rejects([&] { flm::metadata::request_context({{"num_ctx", 8192}, {"options", {{"num_ctx", 16384}}}}); });
    rejects([&] { flm::metadata::request_context({{"options", "invalid"}}); });
    check(flm::metadata::request_context({{"options", nullptr}}) == std::nullopt);
    for (const json invalid : {json(-1), json(0), json(511), json(8192.5), json("8192"), json(true), json(1ULL << 40)})
        rejects([&] { flm::metadata::request_context({{"num_ctx", invalid}}); });
    auto fixed_catalog = catalog;
    fixed_catalog["label"] = {"single-turn"};
    const auto fixed = flm::metadata::describe(fixed_catalog, config, 32768);
    check(!fixed.contains("context_capacity"));
    check(fixed["configured_context_length"] == 8192);
    rejects([&] { flm::metadata::resolve_request_context({{"num_ctx", 8192}}, fixed); });
    const auto vision = flm::metadata::describe({{"vlm", true}, {"details", {{"think", false}}}},
        {{"model_type", "vision"}, {"text_config", {{"model_type", "text"}, {"max_position_embeddings", 131072}}}}, -1);
    check(vision["capabilities"] == json::array({"completion", "vision"}));
    check(vision["model_context_length"] == 131072);
    check(!vision["context_capacity"].contains("default"));
    const auto thinker = flm::metadata::describe(catalog,
        {{"thinker_config", {{"text_config", {{"max_position_embeddings", 131072}}}}}}, -1);
    check(thinker["model_context_length"] == 131072);
    const auto null_type = flm::metadata::describe(catalog,
        {{"model_type", nullptr}, {"max_position_embeddings", 65536}}, -1);
    check(null_type["model_context_length"] == 65536);
    check(!null_type.contains("model_info"));
    const auto speech = flm::metadata::describe({{"asr", true}}, config, -1);
    check(speech["type"] == "audio");
    check(speech["capabilities"] == json::array({"transcription"}));
    check(!speech.contains("context_capacity"));
    check(!speech["context_adjustable"].get<bool>());
    const auto decision = flm::metadata::describe({{"label", {"decision"}}}, config, -1);
    check(decision["capabilities"] == json::array({"decision"}));
    check(!decision.contains("context_capacity"));
    check(flm::metadata::read_config("/does-not-exist").empty());
    std::cout << "FLM metadata and context request tests passed\n";
}