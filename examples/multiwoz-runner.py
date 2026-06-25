import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

BACKEND_ALIASES = {"azure": "azure_openai"}
REQUIRED_CONFIG_KEYS = {
    "model",
    "max_iterations",
    "data_file",
    "output_dir",
    "backend",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    missing = REQUIRED_CONFIG_KEYS - config.keys()
    if missing:
        raise ValueError(f"Config is missing required keys: {sorted(missing)}")
    return config


def make_rlm(config: dict[str, Any], config_path: Path) -> RLM:
    backend = BACKEND_ALIASES.get(str(config["backend"]), str(config["backend"]))
    data_file = resolve_path(str(config["data_file"]), config_path)
    output_dir = resolve_path(str(config["output_dir"]), config_path)
    file_name = f"rlm_{Path(data_file).stem}_{config['model']}_iter{config['max_iterations']}"

    backend_kwargs: dict[str, Any] = {"model_name": str(config["model"])}
    if backend == "azure_openai":
        backend_kwargs.update(
            {
                "api_key": required_env("AZURE_OPENAI_API_KEY"),
                "azure_endpoint": required_env("AZURE_OPENAI_ENDPOINT"),
                "api_version": os.getenv(
                    "AZURE_OPENAI_API_VERSION",
                    str(config.get("api_version", "2024-02-15-preview")),
                ),
            }
        )

    return RLM(
        backend=backend,
        backend_kwargs=backend_kwargs,
        environment=str(config.get("environment", "local")),
        max_iterations=int(config["max_iterations"]),
        max_depth=int(config.get("max_depth", 1)),
        logger=RLMLogger(log_dir=output_dir, file_name=file_name),
        verbose=bool(config.get("verbose", True)),
    )


def resolve_path(value: str, config_path: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return str(path.resolve())


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MultiWOZ RLM evaluation.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    load_dotenv()
    config_path = args.config.resolve()
    config = load_config(config_path)
    data_file = resolve_path(str(config["data_file"]), config_path)
    rlm = make_rlm(config, config_path)

    with open(data_file, encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]

    for record in records:
        result = rlm.completion(
            prompt=record["context"],
            root_prompt=record["question"],
        )

        print(f"\n--- {record['dialogue_id']} ---")
        print(f"Question:        {record['question']}")
        print(f"Model answer:    {result.response}")
        print(f"Expected answer: {record['answer']}")

    print(f"\nDone. Log written to: {rlm.logger.log_file_path}")


if __name__ == "__main__":
    main()
