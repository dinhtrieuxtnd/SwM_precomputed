import argparse
import json
import platform
import random
import torch
import subprocess
import csv

import numpy as np

from pathlib import Path
from datetime import datetime

from src.models.sasrec import SASRec
from src.data.evaluation_dataset import EvaluationDataset
from src.data.train_dataset import TrainDataset
from src.utils.config import load_config, save_config, validate_config
from src.utils.io import create_run_directory, load_pickle, save_pickle
from src.utils.logging import setup_logger
from src.training.loss import SASRecLoss
from src.training.train import train_model
from src.training.evaluate import evaluate_ranking

def valid_processed_data_paths(config: dict) -> dict[str, Path]:
    processed_data_config = config.get("data", {}).get("processed", {})
    if not processed_data_config:
        raise ValueError("Missing 'data.processed' configuration section.")

    output_dir = Path(processed_data_config.get("output_dir", ""))
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Processed data directory '{output_dir}' does not exist.")

    artifacts_dir = output_dir / "artifacts"
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"Artifacts directory '{artifacts_dir}' does not exist in '{output_dir}'.")

    required_files = {
        "train_samples": artifacts_dir / "train_samples.pkl",
        "validation_samples": artifacts_dir / "validation_samples.pkl",
        "test_samples": artifacts_dir / "test_samples.pkl",
        "news_vector_mapping": artifacts_dir / "news_vector_mapping.pkl",
    }

    missing_files = [str(path) for path in required_files.values() if not path.is_file()]
    if missing_files:
        missing_text = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(f"Missing required processed data files:\n{missing_text}")

    return required_files
    

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args()

    config = load_config(args.config)
    
    if args.batch_size is not None:
        if args.batch_size <= 0:
            parser.error("Batch size must be a positive integer.")
        config["training"]["batch_size"] = args.batch_size

    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            parser.error("Learning rate must be a positive float.")
        config["optimizer"]["learning_rate"] = args.learning_rate

    validate_config(config)
    
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        gpu_name = None

    experiment_name = config["experiment"]["name"]

    run_dir = create_run_directory("outputs", experiment_name)
    save_config(config, run_dir / "config.yaml")
    
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "None"
    
    start_time = datetime.now()
    
    run_info = {
        "experiment_name": experiment_name,
        "status": "running",
        "start_time": start_time.isoformat(),
        "python_version": platform.python_version(),
        "seed": config["training"]["seed"],
        "pytorch_version": torch.__version__,
        "device": device,
        "gpu_name": gpu_name,
        "git_commit": git_commit,
    }
    
    with (run_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    logger = setup_logger(run_dir / "train.log")
    logger.info("Device: %s", device)
    if gpu_name:
        logger.info("GPU: %s", gpu_name)
    logger.info("Random seed: %d", seed)
    logger.info("Run directory: %s", run_dir)
    logger.info("Configuration saved")
    
    history_path = run_dir / "history.csv"
    
    with history_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_ndcg@10",
                "learning_rate",
            ]
        )

    try:
        # Sau này đặt quá trình huấn luyện tại đây.
        logger.info("Training started")
        
        required_files = valid_processed_data_paths(config)
        logger.info("Training artifacts verified")
        
        train_samples = load_pickle(required_files["train_samples"])
        val_samples = load_pickle(required_files["validation_samples"])
        news_vector_mapping = load_pickle(required_files["news_vector_mapping"])
        
        logger.info("Loaded processed data: train_samples (%d), validation_samples (%d), news_vector_mapping (%d)",
                    len(train_samples), len(val_samples), len(news_vector_mapping))
        
        padding_id = config["data"]["padding_id"]
        logger.info("Using padding_id: %d", padding_id)
        
        train_dataset = TrainDataset(
            samples=train_samples,
            max_sequence_length=config["data"]["max_sequence_length"],
            padding_id=padding_id,
            mapping=news_vector_mapping,
            vector_size=config["model"]["embedding_dim"],
        )
        logger.info("Train dataset size: %d", len(train_dataset))
        
        
        val_dataset = EvaluationDataset(
            samples=val_samples,
            num_negatives=config["evaluation"]["num_negatives"],
            max_sequence_length=config["data"]["max_sequence_length"],
            padding_id=padding_id,
            mapping=news_vector_mapping,
            vector_size=config["model"]["embedding_dim"],
        )
        
        logger.info("Validation dataset size: %d", len(val_dataset))
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"]["num_workers"],
            pin_memory=device == "cuda",
            persistent_workers=config["training"]["num_workers"] > 0,
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"],
            pin_memory=device == "cuda",
            persistent_workers=config["training"]["num_workers"] > 0,
        )
        logger.info("Data loaders created: train_loader (%d batches), val_loader (%d batches)",
                    len(train_loader), len(val_loader))
        
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu")
        logger.info("Using device: %s", device)
        
        model = SASRec(
            max_sequence_length=config["data"]["max_sequence_length"],
            embedding_dim=config["model"]["embedding_dim"],
            num_blocks=config["model"]["num_blocks"],
            num_heads=config["model"]["num_heads"],
            dropout=config["model"]["dropout"],
        )
        model = model.to(device)
        logger.info("Model architecture:\n%s", model)
        
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Number of trainable parameters: %d", num_parameters)

        criterion = SASRecLoss()
        logger.info("Loss function: %s", criterion)
        
        if config["optimizer"]["name"].lower() == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["optimizer"]["learning_rate"],
            )
            logger.info("Optimizer: AdamW with learning rate %f", config["optimizer"]["learning_rate"])
        else:
            raise ValueError(f"Unsupported optimizer: {config['optimizer']['name']}")

        results = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=config["training"]["epochs"],
            k=config["evaluation"]["k"],
            patience=config["training"]["early_stopping"]["patience"],
            min_delta=config["training"]["early_stopping"]["min_delta"],
            run_dir=run_dir,
            model_config=None,
            max_norm=config["training"]["gradient_clip_norm"]["max_norm"],
            logger=logger,
        )

        # Save training history returned from train_model
        history = results.get("history", {})
        if history:
            save_pickle(history, run_dir / "artifacts" / "training_history.pkl")

            with history_path.open("a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                for i in range(len(history["train_loss"])):
                    writer.writerow([
                        i + 1,
                        history["train_loss"][i],
                        "",  # val_loss not tracked separately yet
                        history["ndcg"][i],
                        config["optimizer"]["learning_rate"],
                    ])
            logger.info("Training history saved to %s and artifacts/training_history.pkl", history_path)

        test_samples = load_pickle(required_files["test_samples"])
        test_dataset = EvaluationDataset(
            samples=test_samples,
            num_negatives=config["evaluation"]["num_negatives"],
            max_sequence_length=config["data"]["max_sequence_length"],
            padding_id=padding_id,
            mapping=news_vector_mapping,
            vector_size=config["model"]["embedding_dim"],
        )
        logger.info("Test dataset size: %d", len(test_dataset))
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"])
        logger.info("Test data loader created: test_loader (%d batches)", len(test_loader))
        
        checkpoint = torch.load(
            results["model_path"],
            map_location=device
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        
        logger.info("Loaded best model from checkpoint: %s", results["model_path"])

        model.to(device)
        model.eval()
        
        logger.info("Evaluating on test dataset...")

        test_metrics = evaluate_ranking(
            model=model,
            dataloader=test_loader,
            device=device,
            k=config["evaluation"]["top_k"],
        )
        
        logger.info("Test metrics: %s", test_metrics)


        end_time = datetime.now()
        run_info["status"] = "completed"
        logger.info("Run completed")
        logger.info(results)

    except Exception as error:
        end_time = datetime.now()
        run_info["status"] = "failed"
        run_info["error"] = str(error)
        logger.exception("Run failed")
        raise

    finally:
        run_info["end_time"] = end_time.isoformat()
        run_info["duration_seconds"] = (end_time - start_time).total_seconds()

        with (run_dir / "run_info.json").open("w", encoding="utf-8") as file:
            json.dump(run_info, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()