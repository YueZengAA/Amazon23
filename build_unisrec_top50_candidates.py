# coding=utf-8
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from numpy_compat import patch_numpy_compat

patch_numpy_compat()

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train UniSRec on All_Beauty, retrieve top-1000 candidates with the two-tower pipeline, "
            "rerank them with full_sort_predict, and keep top-50 per example."
        )
    )
    parser.add_argument("--domain", type=str, default="All_Beauty")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--stopping_step", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--plm_suffix", type=str, default="blair768.feature")
    parser.add_argument("--plm_size", type=int, default=768)
    parser.add_argument("--adaptor_layers", type=str, default="768,300")
    parser.add_argument("--train_stage", type=str, default="inductive_ft")
    parser.add_argument("--ann_k", type=int, default=1000)
    parser.add_argument("--ann_overfetch", type=int, default=256)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--mask_history", dest="mask_history", action="store_true")
    parser.add_argument("--no_mask_history", dest="mask_history", action="store_false")
    parser.set_defaults(mask_history=True)
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="seq_rec_results/dataset/processed/All_Beauty",
    )
    parser.add_argument("--output_dir", type=str, default="artifacts/qwen_rerank")
    return parser.parse_args()


def mask_history_scores(scores: torch.Tensor, hist_ids: torch.Tensor, pos_id: int) -> torch.Tensor:
    hist_ids = hist_ids[hist_ids > 0].unique()
    if hist_ids.numel() > 0:
        scores[hist_ids] = -1e9
    scores[pos_id] = max(float(scores[pos_id].item()), -1e8)
    return scores


def build_interaction(model, batch_rows, item_token2id, device):
    from recbole.data.interaction import Interaction

    seq_lengths = [len(row["history_tokens"]) for row in batch_rows]
    max_len = max(seq_lengths)
    seq_rows = []
    pos_items = []
    for row in batch_rows:
        hist_ids = [item_token2id[token] for token in row["history_tokens"]]
        seq_rows.append([0] * (max_len - len(hist_ids)) + hist_ids)
        pos_items.append(item_token2id[row["pos_item_token"]])
    interaction = Interaction(
        {
            model.ITEM_SEQ: torch.tensor(seq_rows, dtype=torch.long),
            model.ITEM_SEQ_LEN: torch.tensor(seq_lengths, dtype=torch.long),
            model.POS_ITEM_ID: torch.tensor(pos_items, dtype=torch.long),
        }
    )
    return interaction.to(device)


def read_interactions(inter_path):
    rows = []
    with open(inter_path, "r", encoding="utf-8") as fin:
        header = fin.readline()
        if not header:
            return rows
        for line in fin:
            user_token, history_text, pos_item_token = line.rstrip("\n").split("\t")
            rows.append(
                {
                    "user_token": user_token,
                    "history_tokens": [token for token in history_text.split(" ") if token],
                    "pos_item_token": pos_item_token,
                }
            )
    return rows


def build_random_train_candidates(args, dataset, split_rows, output_path):
    item_id2token = dataset.field2id_token["item_id"]
    item_token2id = dataset.field2token_id["item_id"]
    all_item_ids = list(range(1, len(item_id2token)))
    rng = np.random.RandomState(args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for row in split_rows:
            pos_item_id = item_token2id[row["pos_item_token"]]
            seen = {item_token2id[token] for token in row["history_tokens"] if token in item_token2id}
            seen.add(pos_item_id)

            candidates = [pos_item_id]
            while len(candidates) < int(args.topk):
                sampled = int(all_item_ids[rng.randint(len(all_item_ids))])
                if sampled in seen or sampled in candidates:
                    continue
                candidates.append(sampled)

            rng.shuffle(candidates)
            payload = {
                "user_token": row["user_token"],
                "history_tokens": row["history_tokens"],
                "pos_item_token": row["pos_item_token"],
                "candidate_tokens": [item_id2token[item_id] for item_id in candidates],
                "candidate_scores": [],
                "retrieved_1000_tokens": [],
            }
            fout.write(json.dumps(payload) + "\n")

    return {
        "rows": len(split_rows),
        "sampling": "positive_plus_random_negatives",
        "num_candidates": int(args.topk),
        "path": str(output_path),
    }


def train_unisrec(args, root_dir: Path):
    seq_dir = root_dir / "seq_rec_results"
    os.chdir(seq_dir)
    if str(seq_dir) not in sys.path:
        sys.path.insert(0, str(seq_dir))

    from recbole.config import Config
    from recbole.data import data_preparation
    from recbole.utils import get_trainer, init_logger, init_seed
    from seq_rec_results.utils import create_dataset, get_model

    adaptor_layers = [int(x) for x in args.adaptor_layers.split(",") if x.strip()]
    model_class = get_model("UniSRec")
    cfg = Config(
        model=model_class,
        dataset=args.domain,
        config_file_list=["config/overall.yaml", "config/UniSRec.yaml"],
        config_dict={
            "device": args.device,
            "epochs": int(args.epochs),
            "train_batch_size": int(args.train_batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "show_progress": False,
            "stopping_step": int(args.stopping_step),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "plm_suffix": str(args.plm_suffix),
            "plm_size": int(args.plm_size),
            "adaptor_layers": adaptor_layers,
            "train_stage": str(args.train_stage),
            "seed": int(args.seed),
        },
    )
    init_seed(cfg["seed"], cfg["reproducibility"])
    init_logger(cfg)
    dataset = create_dataset(cfg)
    train_data, valid_data, test_data = data_preparation(cfg, dataset)
    model = model_class(cfg, train_data.dataset).to(cfg["device"])
    trainer = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, model)

    best_valid_score = None
    best_valid_result = None
    best_epoch = -1
    best_state = None
    patience = 0
    for epoch in range(int(cfg["epochs"])):
        trainer._train_epoch(train_data, epoch, show_progress=False)
        valid_score, valid_result = trainer._valid_epoch(valid_data, show_progress=False)
        if best_valid_score is None or valid_score > best_valid_score:
            best_valid_score = float(valid_score)
            best_valid_result = dict(valid_result)
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(cfg["stopping_step"]):
                break

    if best_state is None:
        raise RuntimeError("UniSRec training did not produce a best checkpoint.")
    model.load_state_dict(best_state)
    return cfg, dataset, model, {"best_epoch": best_epoch, "best_valid_result": best_valid_result}


@torch.no_grad()
def export_split_candidates(args, model, dataset, split_rows, output_path, device):
    from two_tower_unisrec_blair_eval import build_faiss_ivf_index, get_item_vecs, get_user_vec

    model.eval().to(device)
    item_id2token = dataset.field2id_token[model.ITEM_ID]
    item_token2id = dataset.field2token_id[model.ITEM_ID]

    item_vecs = get_item_vecs(model, device=device)
    use_ann = faiss is not None
    index = None
    item_mat = None
    if use_ann:
        item_mat = item_vecs[1:]
        index = build_faiss_ivf_index(item_mat, nlist=256, nprobe=16)

    batch_size = int(args.eval_batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_hit_1000 = 0
    num_hit_50 = 0
    total = 0
    skipped_not_in_top1000 = 0
    forced_into_top50 = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for start in range(0, len(split_rows), batch_size):
            batch_rows = split_rows[start : start + batch_size]
            interaction = build_interaction(model, batch_rows, item_token2id, device)
            full_scores = model.full_sort_predict(interaction)
            if args.mask_history:
                for i in range(full_scores.size(0)):
                    full_scores[i] = mask_history_scores(
                        full_scores[i], interaction[model.ITEM_SEQ][i], int(interaction[model.POS_ITEM_ID][i].item())
                    )

            if use_ann:
                user_vec, _ = get_user_vec(model, interaction, device=device)
                fetch_k = min(item_mat.shape[0], int(args.ann_k) + int(args.ann_overfetch))
                _, ann_idx = index.search(user_vec.detach().cpu().numpy().astype(np.float32), fetch_k)
            else:
                ann_idx = full_scores.topk(k=min(int(args.ann_k), full_scores.size(1)), dim=1).indices.cpu().numpy()

            for i, row in enumerate(batch_rows):
                pos_item_id = int(interaction[model.POS_ITEM_ID][i].item())
                hist_ids = interaction[model.ITEM_SEQ][i]
                seen = set(int(x) for x in hist_ids.tolist() if int(x) > 0)

                retrieved_ids = []
                for ann_item in ann_idx[i].tolist():
                    item_id = int(ann_item) + 1 if use_ann else int(ann_item)
                    if item_id <= 0 or item_id in seen:
                        continue
                    retrieved_ids.append(item_id)
                    if len(retrieved_ids) >= int(args.ann_k):
                        break

                if pos_item_id in retrieved_ids:
                    num_hit_1000 += 1
                else:
                    skipped_not_in_top1000 += 1
                    continue

                if not retrieved_ids:
                    continue
                cand_tensor = torch.tensor(retrieved_ids, device=full_scores.device, dtype=torch.long)
                cand_scores = full_scores[i].index_select(0, cand_tensor)
                rerank_scores, rerank_order = torch.sort(cand_scores, descending=True)
                rerank_ids = cand_tensor.index_select(0, rerank_order)[: int(args.topk)].tolist()
                rerank_scores = rerank_scores[: int(args.topk)].tolist()
                if pos_item_id not in rerank_ids:
                    pos_rank_in_1000 = retrieved_ids.index(pos_item_id)
                    pos_score = float(cand_scores[pos_rank_in_1000].item())
                    if len(rerank_ids) >= int(args.topk):
                        rerank_ids[-1] = pos_item_id
                        rerank_scores[-1] = pos_score
                    else:
                        rerank_ids.append(pos_item_id)
                        rerank_scores.append(pos_score)
                    forced_into_top50 += 1
                if pos_item_id in rerank_ids:
                    num_hit_50 += 1
                total += 1

                payload = {
                    "user_token": row["user_token"],
                    "history_tokens": row["history_tokens"],
                    "pos_item_token": row["pos_item_token"],
                    "candidate_tokens": [item_id2token[item_id] for item_id in rerank_ids],
                    "candidate_scores": [float(score) for score in rerank_scores],
                    "retrieved_1000_tokens": [item_id2token[item_id] for item_id in retrieved_ids],
                }
                fout.write(json.dumps(payload) + "\n")

    return {
        "rows": total,
        "skipped_not_in_top1000": skipped_not_in_top1000,
        "forced_into_top50": forced_into_top50,
        "cand_hit_rate@1000": num_hit_1000 / max(total, 1),
        "cand_hit_rate@50": num_hit_50 / max(total, 1),
        "path": str(output_path),
    }


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    cfg, dataset, model, train_summary = train_unisrec(args, root_dir)

    output_dir = (root_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "unisrec_best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "domain": args.domain,
                "device": args.device,
                "epochs": args.epochs,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "plm_suffix": args.plm_suffix,
                "plm_size": args.plm_size,
                "adaptor_layers": args.adaptor_layers,
                "train_stage": args.train_stage,
            },
            "summary": train_summary,
        },
        ckpt_path,
    )

    processed_dir = (root_dir / args.processed_dir).resolve()
    splits = {
        "train": read_interactions(str(processed_dir / f"{args.domain}.train.inter")),
        "valid": read_interactions(str(processed_dir / f"{args.domain}.valid.inter")),
        "test": read_interactions(str(processed_dir / f"{args.domain}.test.inter")),
    }

    export_summary = {}
    for split_name, rows in splits.items():
        if split_name == "train":
            export_summary[split_name] = build_random_train_candidates(
                args=args,
                dataset=dataset,
                split_rows=rows,
                output_path=output_dir / f"{split_name}.top{args.topk}.jsonl",
            )
        else:
            export_summary[split_name] = export_split_candidates(
                args=args,
                model=model,
                dataset=dataset,
                split_rows=rows,
                output_path=output_dir / f"{split_name}.top{args.topk}.jsonl",
                device=args.device,
            )
        print(split_name, export_summary[split_name])

    summary = {
        "domain": args.domain,
        "checkpoint": str(ckpt_path),
        "train_summary": train_summary,
        "export_summary": export_summary,
    }
    with open(output_dir / "candidate_summary.json", "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2)
    print("wrote", output_dir / "candidate_summary.json")


if __name__ == "__main__":
    main()
