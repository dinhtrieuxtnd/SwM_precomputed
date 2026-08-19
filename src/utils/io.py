import pickle
import json

from datetime import datetime
from pathlib import Path
from typing import Any


def save_pickle(
    data: Any,
    file_path: str | Path,
) -> None:
    file_path = Path(file_path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open("wb") as file:
        pickle.dump(data, file)


def load_pickle(
    file_path: str | Path,
) -> Any:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with file_path.open("rb") as file:
        return pickle.load(file)
    
def create_run_directory(
    output_root: str | Path,
    experiment_name: str,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_root) / experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    for name in ("checkpoints", "artifacts", "plots", "predictions"):
        (run_dir / name).mkdir()

    return run_dir

def save_run_info(
    run_info: dict[str, Any],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)