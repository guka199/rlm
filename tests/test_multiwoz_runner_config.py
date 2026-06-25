from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_runner_module():
    runner_path = Path(__file__).parents[1] / "examples" / "multiwoz-runner.py"
    spec = importlib.util.spec_from_file_location("multiwoz_runner", runner_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_config_reads_required_container_settings(tmp_path: Path) -> None:
    runner = load_runner_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model: gpt-5.1",
                "max_iterations: 30",
                "data_file: /app/data/multiwoz_qa_30.jsonl",
                "output_dir: /app/logs",
                "backend: azure",
            ]
        ),
        encoding="utf-8",
    )

    config = runner.load_config(config_path)

    assert config["model"] == "gpt-5.1"
    assert config["max_iterations"] == 30
    assert config["backend"] == "azure"


def test_load_config_fails_loudly_for_missing_required_keys(tmp_path: Path) -> None:
    runner = load_runner_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: gpt-5.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required keys"):
        runner.load_config(config_path)


def test_relative_paths_are_resolved_from_config_location(tmp_path: Path) -> None:
    runner = load_runner_module()
    config_path = tmp_path / "config.yaml"

    assert runner.resolve_path("data/input.jsonl", config_path) == str(
        (tmp_path / "data" / "input.jsonl").resolve()
    )


def test_required_env_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = load_runner_module()
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        runner.required_env("AZURE_OPENAI_API_KEY")
