"""A scripts/download_models.py tesztjei (hálózat nélkül, a kéréseket helyettesítjük)."""

import importlib.util
import sys

import pytest
import requests

import config

_spec = importlib.util.spec_from_file_location(
    "download_models", config.ROOT_DIR / "scripts" / "download_models.py"
)
dm = importlib.util.module_from_spec(_spec)
sys.modules["download_models"] = dm  # a dataclass-hoz kell
_spec.loader.exec_module(dm)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_every_hf_model_has_allow_patterns():
    for repo_id in config.HF_MODELS:
        assert dm.HF_ALLOW_PATTERNS.get(repo_id), repo_id


@pytest.mark.parametrize(
    "available, expected",
    [
        (["qwen2.5:1.5b"], True),
        (["qwen2.5:1.5b-instruct-q4_K_M"], True),
        (["qwen2.5:7b", "llama3.2:latest"], False),
        ([], False),
    ],
)
def test_ollama_has_model(available, expected):
    assert dm.ollama_has_model(available, "qwen2.5:1.5b") is expected


def test_ollama_has_model_without_tag():
    assert dm.ollama_has_model(["llama3.2:latest"], "llama3.2") is True


def test_setup_ollama_server_down(monkeypatch):
    def down(*args, **kwargs):
        raise requests.ConnectionError("nem fut")

    monkeypatch.setattr(dm.requests, "get", down)
    result = dm.setup_ollama("http://localhost:11434", "qwen2.5:1.5b", verify=False)
    assert result.ok is False
    assert "nem érhető el" in result.message


def test_setup_ollama_already_present_does_not_pull(monkeypatch):
    monkeypatch.setattr(
        dm.requests, "get", lambda *a, **k: FakeResponse({"models": [{"name": "qwen2.5:1.5b"}]})
    )
    monkeypatch.setattr(dm, "ollama_pull", lambda *a: pytest.fail("nem kellett volna letölteni"))
    result = dm.setup_ollama("http://x", "qwen2.5:1.5b", verify=False)
    assert result.ok is True
    assert result.message == "már letöltve"


def test_setup_ollama_pulls_missing_model(monkeypatch):
    pulled = []
    monkeypatch.setattr(dm.requests, "get", lambda *a, **k: FakeResponse({"models": []}))
    monkeypatch.setattr(dm, "ollama_pull", lambda url, model: pulled.append(model))
    result = dm.setup_ollama("http://x", "qwen2.5:1.5b", verify=False)
    assert result.ok is True
    assert pulled == ["qwen2.5:1.5b"]


def test_parse_args_defaults():
    args = dm.parse_args([])
    assert args.llm == config.LLM_BACKEND
    assert args.no_verify is False


def test_main_reports_failure(monkeypatch, capsys):
    def boom(repo_id):
        raise OSError("nincs internet")

    monkeypatch.setattr(dm, "download_hf_model", boom)
    assert dm.main(["--llm", "none", "--no-verify"]) == 1
    out = capsys.readouterr().out
    assert "[HIBA] 1. modell (ViT): OSError: nincs internet" in out
