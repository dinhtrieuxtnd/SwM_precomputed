import logging
import argparse
import platform

import pandas as pd

from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer

from src.utils.io import save_run_info, save_pickle
from src.data.parser import read_behaviors, read_news
from src.data.mapping import build_combined_mapping, build_news_vector_mapping
from src.utils.config import load_config, save_config
from src.utils.logging import setup_logger
from src.data.preprocessing import build_samples_id, clean_news_text

def validate_input_paths(input_config: dict[str, str]) -> dict[str, Path]:
    input_paths = {
        name: Path(path)
        for name, path in input_config.items()
    }
    
    missing_paths = [
        str(path)
        for path in input_paths.values()
        if not path.is_file()
    ]
    if missing_paths:
        missing_text = "\n".join(
            f"  - {path}" for path in missing_paths
        )
        raise FileNotFoundError(
            f"Missing input files:\n{missing_text}"
        )
    return input_paths

def prepare_output_directory(
    output_config: dict,
    config_path: Path,
) -> Path:
    output_dir = Path(output_config["directory"])
    overwrite = output_config.get("overwrite", False)

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory '{output_dir}' already exists. "
            f"Change output.directory in {config_path}, "
            "or set output.overwrite=true."
        )

    output_dir.mkdir(parents=True, exist_ok=overwrite)
    (output_dir / "artifacts").mkdir(exist_ok=True)

    return output_dir

def log_run_setup(
    logger: logging.Logger,
    preprocess_name: str,
    config_path: Path,
    output_dir: Path,
    input_paths: dict[str, Path],
    start_time: datetime
) -> None:
    logger.info("Starting preprocessing at %s", start_time.isoformat())
    logger.info("Preprocessing: %s", preprocess_name)
    logger.info("Config: %s", config_path)
    logger.info("Output directory: %s", output_dir)
    
    for name, path in input_paths.items():
        logger.info("Input %s: %s", name, path)
        
    logger.info("Saved config to: %s", output_dir / "config.yaml")
    
def split_dev_by_time(
    dev_df: pd.DataFrame,
    validation_ratio: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    
    dev_df = dev_df.copy()
    dev_df["time"] = pd.to_datetime(
        dev_df["time"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="raise",
    )

    dev_df_sorted = dev_df.sort_values(
        by=["time", "impression_id"],
    ).reset_index(drop=True)
    
    split_index = int(len(dev_df_sorted) * validation_ratio)

    validation_df = dev_df_sorted.iloc[:split_index]
    test_df = dev_df_sorted.iloc[split_index:]

    return validation_df, test_df

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    
    start_time = datetime.now()

    config = load_config(args.config)
    
    preprocess_name = config["preprocessing"]["name"]
    input_paths = validate_input_paths(config["input"])
    output_dir = prepare_output_directory(config["output"], args.config)
    embedding_model = config["embedding"]["model"]
    
    save_config(config, output_dir / "config.yaml")
    
    logger = setup_logger(output_dir / "preprocess.log")
    
    run_info = {
        "preprocess_name": preprocess_name,
        "status": "running",
        "start_time": start_time.isoformat(),
        "python_version": platform.python_version(),
        "seed": config["preprocessing"]["seed"],
        "embedding_model": embedding_model,
    }
    
    save_run_info(run_info, output_dir / "run_info.json")
    log_run_setup(
        logger=logger,
        preprocess_name=preprocess_name,
        config_path=args.config,
        output_dir=output_dir,
        input_paths=input_paths,
        start_time=start_time
    )
    
    try:
        train_behaviors: pd.DataFrame = read_behaviors(input_paths["train_behaviors"])
        logger.info("Train behaviors shape: %s", train_behaviors.shape)
        
        dev_behaviors: pd.DataFrame = read_behaviors(input_paths["dev_behaviors"])
        logger.info("Dev behaviors shape: %s", dev_behaviors.shape)

        validation_df, test_df = split_dev_by_time(
            dev_df=dev_behaviors,
            validation_ratio=config["split"]["validation_ratio"]
        )
        
        logger.info("Validation behaviors shape: %s", validation_df.shape)
        logger.info("Test behaviors shape: %s", test_df.shape)
        
        behavior_train_samples_id: list[dict] = build_samples_id(train_behaviors)
        behavior_val_samples_id: list[dict] = build_samples_id(validation_df)
        behavior_test_samples_id: list[dict] = build_samples_id(test_df)
        
        logger.info("Train samples count: %d", len(behavior_train_samples_id))
        logger.info("Validation samples count: %d", len(behavior_val_samples_id))
        logger.info("Test samples count: %d", len(behavior_test_samples_id))
        
        train_news_df: pd.DataFrame = read_news(input_paths["train_news"])
        logger.info("Train news shape: %s", train_news_df.shape)
        
        dev_news_df: pd.DataFrame = read_news(input_paths["dev_news"])
        logger.info("Dev news shape: %s", dev_news_df.shape)
        
        columns_used_for_mapping = config["text"]["columns"]
        
        cleaned_news_train_df = clean_news_text(
            df=train_news_df,
            columns=columns_used_for_mapping
        )
        cleaned_news_dev_df = clean_news_text(
            df=dev_news_df,
            columns=columns_used_for_mapping
        )
        logger.info("Cleaned news!")
        
        news_mapping = build_combined_mapping(
            train_news_df=cleaned_news_train_df,
            dev_news_df=cleaned_news_dev_df,
            column_names=columns_used_for_mapping
        )
        logger.info("Built news mapping!")

        model = SentenceTransformer(embedding_model)
        model.max_seq_length = config["embedding"]["max_token_length"]
        
        news_vector_mapping = build_news_vector_mapping(
            mapping=news_mapping,
            model=model
        )
        
        logger.info("Built news vector mapping!")

        metadata = {
            "train_samples_count": len(behavior_train_samples_id),
            "validation_samples_count": len(behavior_val_samples_id),
            "test_samples_count": len(behavior_test_samples_id),
            "max_token_length": config["embedding"]["max_token_length"],
            "embedding_model": embedding_model,
        }
        
        artifacts = {
            "train_samples.pkl": behavior_train_samples_id,
            "validation_samples.pkl": behavior_val_samples_id,
            "test_samples.pkl": behavior_test_samples_id,
            "news_vector_mapping.pkl": news_vector_mapping,
        }

        for filename, data in artifacts.items():
            save_pickle(data, output_dir / "artifacts" / filename)
            logger.info("Saved artifact: %s", filename)

        save_run_info(
            metadata,
            output_dir / "statistics.json",
        )
        logger.info("Saved statistics.json")
        
        end_time = datetime.now()
        run_info["status"] = "completed"
        logger.info("Preprocessing completed")
    
    except Exception as e:
        end_time = datetime.now()
        run_info["status"] = "failed"
        run_info["error"] = str(e)
        logger.exception("Preprocessing failed: %s", e)
        raise

    finally:
        run_info["end_time"] = end_time.isoformat()
        run_info["duration_seconds"] = (end_time - start_time).total_seconds()
        
        save_run_info(run_info, output_dir / "run_info.json")
        

if __name__ == "__main__":
    main()