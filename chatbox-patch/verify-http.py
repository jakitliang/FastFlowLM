"""Read-only FLM patch checks; never requests generation, downloads, or model switches."""
import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="Test server URL without /v1")
    parser.add_argument("--model", help="Optional exact catalog model ID for /api/show")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as response:
            return json.load(response)

    models = request("/v1/models")["data"]
    for model in models:
        assert "details" in model and isinstance(model["capabilities"], list), model
        if "context_capacity" in model:
            capacity = model["context_capacity"]
            assert capacity["source"] == "installed_config", model
            assert 0 < capacity["max"] <= model["model_context_length"], model
            assert capacity["request_parameter"] == "num_ctx", model
    running = request("/api/ps")["models"]
    for model in running:
        assert model["name"] != "model-faker", model
        assert isinstance(model["context_length"], int) and model["context_length"] > 0, model
        assert "expires_at" not in model, model
    result = {"catalog_count": len(models), "running": running}
    if args.model:
        result["show"] = request("/api/show", {"model": args.model})
        assert result["show"]["model"] == args.model
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()