import json

from yabot import tokens


def test_context_window_uses_models_json(tmp_path, monkeypatch):
    data = {"data": [{"id": "gpt-4o-mini", "context_length": 12345}]}
    path = tmp_path / "models.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(tokens, "MODELS_DATA_PATH", path)
    tokens._reset_models_context_cache_for_tests()

    assert tokens.context_window_for_model("gpt-4o-mini") == 12345
