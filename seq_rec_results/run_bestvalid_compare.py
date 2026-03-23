import argparse
import json
import os
import sys

import torch
import numpy as np

from numpy_compat import patch_numpy_compat

patch_numpy_compat()

from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import get_trainer, init_logger, init_seed


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train a seq-rec model, keep best-valid weights in memory, and evaluate test on best-valid and last epoch."
        )
    )
    p.add_argument("--model", type=str, default="UniSRec", help="RecBole model name (e.g., UniSRec, SASRecText)")
    p.add_argument("--dataset", type=str, default="All_Beauty")
    p.add_argument("--config_overall", type=str, default="config/overall.yaml")
    p.add_argument("--config_model", type=str, default=None, help="Model config yaml, default config/<model>.yaml")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train_batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=2048)
    p.add_argument("--stopping_step", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--plm_size", type=int, required=True)
    p.add_argument("--plm_suffix", type=str, required=True, help="Feature suffix, e.g. blair768.feature")
    p.add_argument(
        "--adaptor_layers",
        type=str,
        default=None,
        help="Comma-separated ints, e.g. 768,64. If omitted, keep config default.",
    )
    p.add_argument(
        "--out_json",
        type=str,
        required=True,
        help="Path to write results JSON (relative to seq_rec_results/)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    # Ensure seq_rec_results/ is on sys.path for utils.py
    sys.path.insert(0, os.getcwd())

    from utils import create_dataset, get_model  # local

    config_model = args.config_model or f"config/{args.model}.yaml"

    overrides = {
        "device": args.device,
        "epochs": int(args.epochs),
        "train_batch_size": int(args.train_batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "show_progress": False,
        "stopping_step": int(args.stopping_step),
        "plm_size": int(args.plm_size),
        "plm_suffix": str(args.plm_suffix),
    }
    if args.learning_rate is not None:
        overrides["learning_rate"] = float(args.learning_rate)
    if args.weight_decay is not None:
        overrides["weight_decay"] = float(args.weight_decay)
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.adaptor_layers:
        overrides["adaptor_layers"] = [int(x) for x in args.adaptor_layers.split(",") if x.strip()]

    model_class = get_model(args.model)
    config = Config(
        model=model_class,
        dataset=args.dataset,
        config_file_list=[args.config_overall, config_model],
        config_dict=overrides,
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    ds = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, ds)

    model = model_class(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    best_valid_score = None
    best_valid_result = None
    best_epoch = -1
    best_state = None

    for epoch in range(int(config["epochs"])):
        trainer._train_epoch(train_data, epoch, show_progress=config["show_progress"])
        valid_score, valid_result = trainer._valid_epoch(valid_data, show_progress=config["show_progress"])

        if best_valid_score is None or valid_score > best_valid_score:
            best_valid_score = float(valid_score)
            best_valid_result = valid_result
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    test_last = trainer.evaluate(test_data, load_best_model=False, show_progress=config["show_progress"])
    if best_state is not None:
        model.load_state_dict(best_state)
    test_best = trainer.evaluate(test_data, load_best_model=False, show_progress=config["show_progress"])

    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "plm_size": int(args.plm_size),
        "plm_suffix": str(args.plm_suffix),
        "device": str(args.device),
        "epochs": int(config["epochs"]),
        "best_epoch": int(best_epoch),
        "best_valid_score": float(best_valid_score) if best_valid_score is not None else None,
        "best_valid": dict(best_valid_result) if best_valid_result is not None else None,
        "test_best_valid": dict(test_best),
        "test_last": dict(test_last),
    }

    out_path = args.out_json
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("best_valid", payload["best_valid"], "epoch", best_epoch)
    print("test(best_valid)", payload["test_best_valid"])
    print("test(last)", payload["test_last"])
    print("wrote", out_path)


if __name__ == "__main__":
    main()
