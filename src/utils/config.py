from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} is not a valid YAML dictionary.")
        
    return config

def save_config(
    config: dict[str, Any],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True,
        )
        
def validate_config(config: dict[str, Any]) -> None:
    training = config.get("training")

    if not isinstance(training, dict):
        raise ValueError("Thiếu mục 'training' trong config")

    if training.get("epochs", 0) <= 0:
        raise ValueError("training.epochs phải lớn hơn 0")

    if training.get("batch_size", 0) <= 0:
        raise ValueError("training.batch_size phải lớn hơn 0")

    if training.get("seed") is None:
        raise ValueError("Thiếu training.seed")