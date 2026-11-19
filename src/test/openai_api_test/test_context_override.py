#!/usr/bin/env python3
"""Per-request context overrides on OpenAI-compatible and native chat."""

from __future__ import annotations

from typing import Any

import flm_api
from flm_api import Api, Skip, case, expect
from test_http_safety import call, expect_400


@case
def invalid_num_ctx_is_rejected_before_model_load(api: Api) -> None:
    for path, param in (("/v1/chat/completions", "num_ctx"), ("/api/chat", "options.num_ctx")):
        for value in (None, True, "512", 512.0, 0, -1, 2**31, 2**64, [], {}):
            body: dict[str, Any] = {"model": "not-a-real-fastflow-model:0b", "messages": [], "stream": True}
            if param == "num_ctx":
                body["num_ctx"] = value
            else:
                body["options"] = {"num_ctx": value}
            status, _, reply = call(api, "POST", path, body)
            expect_400(status, reply, param, f"{path}: {value!r}")


@case
def override_loads_reuses_reports_and_reverts(api: Api) -> None:
    status, _, listing = call(api, "GET", "/v1/models?installed=true&include=capabilities")
    expect(status == 200, f"model capability listing: {status} {listing!r:.200}")
    model = next((entry for entry in listing["data"] if entry["id"] == api.model), None)
    if model is None:
        raise Skip("model is not installed")
    caps = model.get("flm", {})
    status, _, plain = call(api, "GET", "/v1/models?installed=true")
    expect(status == 200 and all("flm" not in entry for entry in plain["data"]),
           "capability must be opt-in")
    if "context_window_override" not in caps:
        status, _, reply = call(api, "POST", "/v1/chat/completions",
                                {"model": api.model, "messages": [], "num_ctx": 1024})
        expect_400(status, reply, "num_ctx", "unsupported model override")
        return
    expect(caps["context_window_override"] is True, f"unexpected context override capability: {model}")

    def chat(*, native: bool = False, num_ctx: int | None = None) -> dict[str, Any]:
        path = "/api/chat" if native else "/v1/chat/completions"
        body: dict[str, Any] = {
            "model": api.model, "messages": [{"role": "user", "content": "Say hello."}],
            "stream": False, "think": False,
        }
        fields = body.setdefault("options", {}) if native else body
        fields["num_predict" if native else "max_tokens"] = 4
        if num_ctx is not None:
            fields["num_ctx"] = num_ctx
        status, _, reply = call(api, "POST", path, body, timeout=api.timeout)
        expect(status == 200 and "error" not in reply, f"{path}: {status} {reply!r:.300}")
        count = reply["eval_count"] if native else reply["usage"]["completion_tokens"]
        expect(count <= 4, f"{path}: output budget exceeded ({count} tokens)")
        status, _, ps = call(api, "GET", "/api/ps")
        expect(status == 200 and ps.get("models"), f"empty or failed /api/ps: {ps!r:.300}")
        loaded = ps["models"][0]
        expect(loaded["model"] == api.model, f"unexpected loaded model: {loaded}")
        if num_ctx is None:
            expect("requested_context_length" not in loaded, f"explicit override leaked: {loaded}")
        else:
            expect(loaded.get("requested_context_length") == num_ctx, f"requested context lost: {loaded}")
        return loaded

    default = chat()["context_length"]
    maximum = caps.get("max_position_embeddings")
    if isinstance(maximum, int) and maximum < 2**31 - 1:
        for path, fields, param in (
            ("/v1/chat/completions", {"num_ctx": maximum + 1}, "num_ctx"),
            ("/api/chat", {"options": {"num_ctx": maximum + 1}}, "options.num_ctx"),
        ):
            status, _, reply = call(api, "POST", path, {"model": api.model, "messages": [], **fields})
            expect_400(status, reply, param, f"{path}: override above model limit")
        status, _, ps = call(api, "GET", "/api/ps")
        expect(status == 200 and ps["models"][0]["context_length"] == default,
               "rejecting an oversized override disturbed the loaded model")
    try:
        first = chat(num_ctx=1024)
        expect(first["context_length"] == 1024, f"first override not loaded: {first}")
        expect(chat(native=True, num_ctx=1024)["context_length"] == 1024,
               "matching native and OpenAI overrides differ")
        clamped = chat(native=True, num_ctx=1)
        expect(clamped["context_length"] == 512, f"512-token floor not applied: {clamped}")
        expect(chat(num_ctx=1)["context_length"] == 512, "repeated clamped request gives different loaded context")
        expect(chat(native=True)["context_length"] == default, "omitting override did not restore default")
    finally:
        restored = chat()
        expect(restored["context_length"] == default, f"original context was not restored: {restored}")


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
