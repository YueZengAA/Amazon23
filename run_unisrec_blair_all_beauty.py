import os
import sys
from pathlib import Path
import subprocess


def _get_env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


def main() -> int:
    root = Path(__file__).resolve().parent
    seq_dir = root / "seq_rec_results"
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {seq_dir}")

    # Defaults match the parameters discussed in chat.
    # You can override any of these with env vars:
    #   DEVICE, EPOCHS, TRAIN_BS, EVAL_BS, LR, WD, DOMAIN
    domain = _get_env("DOMAIN", "All_Beauty")
    # Paper's sequential recommendation uses all interactions (no 5-core filtering).
    core = _get_env("CORE", "0core")  # 0core or 5core
    device = _get_env("DEVICE", "cuda")
    epochs = int(_get_env("EPOCHS", "30"))
    train_bs = int(_get_env("TRAIN_BS", "1024"))
    eval_bs = int(_get_env("EVAL_BS", "4096"))
    lr = float(_get_env("LR", "0.0003"))
    wd = float(_get_env("WD", "0.00001"))

    # BLaIR feature settings
    plm_size = 768
    plm_suffix = "blair768.feature"
    adaptor_layers = "768,300"

    out_json = str((root / "results" / f"{domain}.blair.json").resolve())

    # Make sure the minimal required files exist. If not, build them.
    data_root = root / "data" / "amazon_reviews_2023"
    in_dir = data_root / "benchmark" / core / "timestamp_w_his"
    meta_jsonl = data_root / "raw" / "meta_categories" / f"meta_{domain}.jsonl"
    processed_dir = root / "seq_rec_results" / "dataset" / "processed" / domain
    feat_path = processed_dir / f"{domain}.blair768.feature"

    # 0) CUDA availability check
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                print("WARNING: DEVICE=cuda requested but torch.cuda.is_available()=False; falling back to cpu")
                device = "cpu"
        except Exception:
            print("WARNING: torch not importable; falling back to cpu")
            device = "cpu"

    # 1) Download raw files if missing
    if not (in_dir / f"{domain}.train.csv").exists() or not meta_jsonl.exists():
        try:
            from huggingface_hub import snapshot_download
        except Exception as e:
            raise RuntimeError("Missing huggingface_hub. Install it in your venv.") from e

        allow_patterns = [
            f"benchmark/{core}/timestamp_w_his/{domain}.*.csv",
            f"raw/meta_categories/meta_{domain}.jsonl",
        ]
        print("Downloading required files from HF Hub...")
        snapshot_download(
            repo_id="McAuley-Lab/Amazon-Reviews-2023",
            repo_type="dataset",
            local_dir=str(data_root),
            allow_patterns=allow_patterns,
        )

    # 2) Build RecBole dataset (.inter + data_maps) if missing
    train_inter = processed_dir / f"{domain}.train.inter"
    maps_path = processed_dir / f"{domain}.data_maps"
    if not train_inter.exists() or not maps_path.exists():
        print("Building RecBole dataset files...")
        subprocess.check_call(
            [
                sys.executable,
                str(root / "seq_rec_results" / "dataset" / "build_from_timestamp_w_his_csv.py"),
                "--domain",
                domain,
                "--input_dir",
                str(in_dir),
                "--output_dir",
                str(root / "seq_rec_results" / "dataset" / "processed"),
            ]
        )

    # 3) Generate BLaIR item features if missing
    if not feat_path.exists():
        print("Generating BLaIR item features...")
        subprocess.check_call(
            [
                sys.executable,
                str(root / "seq_rec_results" / "dataset" / "gen_item_feature_blair_from_meta_jsonl.py"),
                "--domain",
                domain,
                "--data_dir",
                str(root / "seq_rec_results" / "dataset" / "processed"),
                "--meta_jsonl",
                str(meta_jsonl),
                "--device",
                device,
                "--max_length",
                "64",
                "--pooler",
                "cls",
            ]
        )

    os.chdir(seq_dir)

    # Ensure local module import works.
    sys.path.insert(0, str(seq_dir))

    import run_bestvalid_compare  # local import

    argv = [
        "run_bestvalid_compare.py",
        "--model",
        "UniSRec",
        "--dataset",
        domain,
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--train_batch_size",
        str(train_bs),
        "--eval_batch_size",
        str(eval_bs),
        "--learning_rate",
        str(lr),
        "--weight_decay",
        str(wd),
        "--plm_size",
        str(plm_size),
        "--plm_suffix",
        plm_suffix,
        "--adaptor_layers",
        adaptor_layers,
        "--out_json",
        out_json,
    ]

    sys.argv = argv
    run_bestvalid_compare.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
