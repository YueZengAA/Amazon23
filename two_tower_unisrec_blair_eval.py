import argparse
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import torch

from numpy_compat import patch_numpy_compat

patch_numpy_compat()

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train UniSRec as a user tower (behavior encoder) and evaluate dot-product retrieval against "
            "BLaIR item embeddings (item tower)."
        )
    )
    p.add_argument("--domain", type=str, default="All_Beauty")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train_batch_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=2048)
    p.add_argument("--stopping_step", type=int, default=10)

    p.add_argument("--plm_suffix", type=str, default="blair768.feature")
    p.add_argument("--plm_size", type=int, default=768)
    p.add_argument("--adaptor_layers", type=str, default="768,300")
    p.add_argument("--train_stage", type=str, default="inductive_ft", choices=["pretrain", "inductive_ft", "transductive_ft"])

    p.add_argument("--mask_history", action="store_true", help="Mask history items when ranking (more realistic).")
    p.add_argument("--max_eval_batches", type=int, default=0, help="0 means all")
    p.add_argument(
        "--ann_k",
        type=int,
        default=5000,
        help="ANN retrieval size (candidates per user).",
    )
    p.add_argument(
        "--ann_overfetch",
        type=int,
        default=256,
        help="Extra items to fetch before masking history.",
    )
    p.add_argument(
        "--use_ann",
        action="store_true",
        help="Use FAISS retrieval instead of full-sort scoring.",
    )
    p.add_argument(
        "--ivf_nlist",
        type=int,
        default=256,
        help="Number of IVF clusters (nlist).",
    )
    p.add_argument(
        "--ivf_nprobe",
        type=int,
        default=16,
        help="Number of IVF clusters to probe at query time.",
    )

    # Precomputed feature mode: skip training entirely, load exported .npy files.
    p.add_argument(
        "--precomputed",
        action="store_true",
        help=(
            "Skip training. Load user_feature.npy and combined_item_feature.npy from "
            "--precomputed_dir, build a FAISS index over item vectors, and evaluate "
            "recall@K for all K in --recall_ks."
        ),
    )
    p.add_argument(
        "--precomputed_dir",
        type=str,
        default="seq_rec_results/dataset/processed/All_Beauty",
        help="Directory containing user_feature.npy, combined_item_feature.npy, pos_items.npy (optional).",
    )
    p.add_argument(
        "--pos_items_file",
        type=str,
        default="",
        help="Path to pos_items.npy (int64 [N]). If empty, loaded from --precomputed_dir/pos_items.npy.",
    )
    p.add_argument(
        "--recall_ks",
        type=str,
        default="1000,2000,5000",
        help="Comma-separated list of K values for recall@K evaluation.",
    )
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(topk_idx: torch.Tensor, pos_items: torch.Tensor, ks=(10, 50)):
    # topk_idx: [B, Kmax], pos_items: [B]
    res = {}
    B = topk_idx.size(0)
    for k in ks:
        top = topk_idx[:, :k]
        hit = (top == pos_items.unsqueeze(1)).any(dim=1).float()
        recall = hit.mean().item()
        # ndcg: 1/log2(rank+1) if hit else 0
        ndcg = torch.zeros_like(hit)
        # find rank positions
        eq = top == pos_items.unsqueeze(1)
        if eq.any():
            ranks = torch.argmax(eq.int(), dim=1) + 1  # 1-based, arbitrary when not hit
            ndcg = torch.where(hit > 0, 1.0 / torch.log2(ranks.float() + 1.0), ndcg)
        res[f"recall@{k}"] = recall
        res[f"ndcg@{k}"] = ndcg.mean().item()
    return res


def cand_hit_rate_from_topk(topk_idx: torch.Tensor, pos_items: torch.Tensor) -> float:
    # topk_idx: [B, K]
    hit = (topk_idx == pos_items.unsqueeze(1)).any(dim=1).float().mean().item()
    return float(hit)


def build_faiss_ivf_index(item_vecs: np.ndarray, nlist: int, nprobe: int):
    if faiss is None:
        raise RuntimeError("faiss is not installed. Install faiss-cpu in your venv.")
    if item_vecs.dtype != np.float32:
        item_vecs = item_vecs.astype(np.float32)
    dim = item_vecs.shape[1]
    n_items = item_vecs.shape[0]
    nlist = max(1, min(int(nlist), n_items))
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(item_vecs)
    index.add(item_vecs)
    index.nprobe = max(1, min(int(nprobe), nlist))
    return index


@torch.no_grad()
def get_user_vec(model, interaction, device: str):
    interaction = interaction.to(device)
    item_seq = interaction[model.ITEM_SEQ]
    item_seq_len = interaction[model.ITEM_SEQ_LEN]
    x_img = model._img_seq(item_seq) if hasattr(model, '_img_seq') else None
    x_review = model._review_seq(item_seq) if hasattr(model, '_review_seq') else None
    item_emb_list = model.moe_adaptor(model.plm_embedding(item_seq), x_img, x_review)
    seq_output = model.forward(item_seq, item_emb_list, item_seq_len)
    seq_output = torch.nn.functional.normalize(seq_output, dim=-1)
    return seq_output, interaction


@torch.no_grad()
def get_item_vecs(model, device: str):
    model.eval()
    model.to(device)
    img_w = model.img_embedding.weight if getattr(model, 'img_embedding', None) is not None else None
    review_w = model.review_embedding.weight if getattr(model, 'review_embedding', None) is not None else None
    item_vec = model.moe_adaptor(model.plm_embedding.weight, img_w, review_w)
    if getattr(model, "train_stage", None) == "transductive_ft":
        item_vec = item_vec + model.item_embedding.weight
    item_vec = torch.nn.functional.normalize(item_vec, dim=-1)
    return item_vec.detach().cpu().numpy().astype(np.float32)


def filter_history(cands: List[int], hist_ids: torch.Tensor, pos_id: int, k: int) -> List[int]:
    seen = set(int(x) for x in hist_ids.tolist() if int(x) > 0)
    out = []
    for it in cands:
        if it in seen:
            continue
        out.append(it)
        if len(out) >= k:
            break
    # Ensure positive isn't dropped (if it was in history)
    if pos_id not in out and pos_id not in seen:
        pass
    return out


@torch.no_grad()
def evaluate_two_tower(model, test_data, mask_history: bool, device: str, max_eval_batches: int = 0):
    model.eval()
    model.to(device)
    KMAX = 50
    CAND_K = 5000
    all_rows = []
    cand_hits = []

    for b, batch in enumerate(test_data):
        if max_eval_batches and b >= max_eval_batches:
            break

        # RecBole dataloaders sometimes yield a tuple like (interaction, idx).
        interaction = batch[0] if isinstance(batch, tuple) else batch
        interaction = interaction.to(device)
        scores = model.full_sort_predict(interaction)  # [B, n_items]

        # Optionally mask history items
        if mask_history:
            item_seq = interaction[model.ITEM_SEQ]  # [B, L]
            pos = interaction[model.POS_ITEM_ID]  # [B]
            # mask all ids in sequence (excluding PAD=0)
            for i in range(item_seq.size(0)):
                ids = item_seq[i]
                ids = ids[ids > 0].unique()
                if ids.numel() > 0:
                    scores[i, ids] = -1e9
                # restore positive in case it appeared in history
                scores[i, pos[i]] = scores[i, pos[i]].clamp(min=-1e8)

        # Candidate coverage@5000
        cand_scores, cand_idx = torch.topk(scores, k=min(CAND_K, scores.size(1)), dim=1)
        # Metrics@10/50
        top_idx = cand_idx[:, :KMAX]
        pos_items = interaction[model.POS_ITEM_ID]
        all_rows.append(compute_metrics(top_idx, pos_items, ks=(10, 50)))
        cand_hits.append(cand_hit_rate_from_topk(cand_idx, pos_items))

    # aggregate
    if not all_rows:
        return {}
    keys = all_rows[0].keys()
    out = {k: float(np.mean([r[k] for r in all_rows])) for k in keys}
    out["cand_hit_rate@5000"] = float(np.mean(cand_hits)) if cand_hits else 0.0
    return out


@torch.no_grad()
def evaluate_ann_retrieval(
    model,
    test_data,
    ann_k: int,
    overfetch: int,
    mask_history: bool,
    device: str,
    max_eval_batches: int = 0,
    ivf_nlist: int = 256,
    ivf_nprobe: int = 16,
):
    model.eval()
    model.to(device)
    KMAX = 50
    ann_k = int(ann_k)
    overfetch = int(overfetch)

    # Build item index from adapted item vectors (exclude PAD=0)
    item_vecs = get_item_vecs(model, device=device)  # [n_items, H]
    item_mat = item_vecs[1:]  # faiss ids 0..n-2 -> item_id = id+1
    index = build_faiss_ivf_index(item_mat, nlist=ivf_nlist, nprobe=ivf_nprobe)

    rows = []
    hit_at_ann = []

    for b, batch in enumerate(test_data):
        if max_eval_batches and b >= max_eval_batches:
            break
        interaction = batch[0] if isinstance(batch, tuple) else batch
        user_vec, interaction = get_user_vec(model, interaction, device=device)  # [B,H]
        pos_items = interaction[model.POS_ITEM_ID]

        # Retrieve
        fetch_k = min(item_mat.shape[0], ann_k + max(overfetch, 0))
        scores, idx = index.search(user_vec.detach().cpu().numpy().astype(np.float32), fetch_k)
        # idx -> item_id
        cand_ids = (idx + 1).tolist()

        # Build per-example ranked list, apply history mask if requested
        item_seq = interaction[model.ITEM_SEQ]
        for i in range(len(cand_ids)):
            ranked = [int(x) for x in cand_ids[i] if int(x) > 0]
            if mask_history:
                ranked = filter_history(ranked, item_seq[i], int(pos_items[i].item()), ann_k)
            else:
                ranked = ranked[:ann_k]

            # candidate hit rate@ann_k
            hit_at_ann.append(1.0 if int(pos_items[i].item()) in ranked[:ann_k] else 0.0)
            # metrics@10/50 from retrieved order
            top50 = ranked[:KMAX]
            top = torch.tensor(top50 + [0] * max(0, KMAX - len(top50)), dtype=torch.long).unsqueeze(0)
            rows.append(compute_metrics(top, pos_items[i : i + 1].cpu(), ks=(10, 50)))

    out = {}
    if rows:
        keys = rows[0].keys()
        out.update({k: float(np.mean([r[k] for r in rows])) for k in keys})
    out[f"cand_hit_rate@{ann_k}"] = float(np.mean(hit_at_ann)) if hit_at_ann else 0.0
    return out


def evaluate_precomputed(
    user_feature: np.ndarray,        # [N, H]  normalised user vecs
    item_feature: np.ndarray,        # [H, n_items]  transposed normalised item vecs
    pos_items: np.ndarray,           # [N]  int64 ground-truth item ids (1-based)
    recall_ks: List[int],
    ivf_nlist: int,
    ivf_nprobe: int,
    mask_history: bool = False,
    item_seqs: np.ndarray = None,    # [N, L] int64, required when mask_history=True
) -> dict:
    """
    ANN recall evaluation using precomputed user/item embeddings.
    item_feature is [H, n_items] (transposed); index built over item_feature.T[1:].
    FAISS id i → item_id = i+1 (skip PAD row 0).
    """
    if faiss is None:
        raise RuntimeError("faiss not installed. Run: pip install faiss-cpu")

    max_k = max(recall_ks)
    # item_feature is [H, n_items]; transpose back to [n_items, H]
    item_mat = item_feature.T  # [n_items, H]
    mat = item_mat[1:].astype(np.float32)  # skip PAD=0; faiss id i → item_id = i+1

    print(
        f"Building FAISS IVF index over {mat.shape[0]} items (dim={mat.shape[1]}, "
        f"nlist={min(max(1, int(ivf_nlist)), mat.shape[0])}, nprobe={int(ivf_nprobe)}) ..."
    )
    index = build_faiss_ivf_index(mat, nlist=ivf_nlist, nprobe=ivf_nprobe)

    overfetch = 256
    fetch_k = min(mat.shape[0], max_k + overfetch)

    # Accumulators per K
    hits = {k: [] for k in recall_ks}

    N = user_feature.shape[0]
    BATCH = 512
    for start in range(0, N, BATCH):
        u_batch = user_feature[start : start + BATCH].astype(np.float32)  # [B, H]
        _, idx_batch = index.search(u_batch, fetch_k)                      # [B, fetch_k]

        for bi in range(u_batch.shape[0]):
            gi = start + bi
            pos = int(pos_items[gi])
            cand = [int(x) + 1 for x in idx_batch[bi] if int(x) >= 0]    # item_id = faiss_id+1

            if mask_history and item_seqs is not None:
                hist = set(int(x) for x in item_seqs[gi] if int(x) > 0)
                filtered = []
                for it in cand:
                    if it not in hist:
                        filtered.append(it)
                cand = filtered

            for k in recall_ks:
                top_k = cand[:k]
                hits[k].append(1.0 if pos in top_k else 0.0)

    return {f"recall@{k}": float(np.mean(hits[k])) for k in recall_ks}


def main():
    args = parse_args()
    set_seed(args.seed)

    # Work in seq_rec_results/ so local imports resolve.
    root = Path(__file__).resolve().parent
    seq_dir = root / "seq_rec_results"
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Missing: {seq_dir}")
    os.chdir(seq_dir)
    sys.path.insert(0, str(seq_dir))

    # ── Precomputed mode: no training needed ────────────────────────────────
    if args.precomputed:
        precomputed_dir = Path(args.precomputed_dir)
        if not precomputed_dir.is_absolute():
            precomputed_dir = (root / precomputed_dir).resolve()

        user_feature = np.load(str(precomputed_dir / "user_feature.npy"))
        item_feature = np.load(str(precomputed_dir / "combined_item_feature.npy"))
        print(f"user_feature:         {user_feature.shape}")
        print(f"combined_item_feature: {item_feature.shape}")

        # pos_items: try explicit path, then default
        pos_items_path = args.pos_items_file if args.pos_items_file else str(precomputed_dir / "pos_items.npy")
        if not Path(pos_items_path).exists():
            raise FileNotFoundError(
                f"pos_items.npy not found at {pos_items_path}. "
                "Run export_and_verify.py first, or pass --pos_items_file."
            )
        pos_items = np.load(pos_items_path).astype(np.int64)
        print(f"pos_items:            {pos_items.shape}")

        # item_seqs for history masking (optional)
        item_seqs = None
        item_seqs_path = precomputed_dir / "item_seq.npy"
        if args.mask_history:
            if item_seqs_path.exists():
                item_seqs = np.load(str(item_seqs_path)).astype(np.int64)
                print(f"item_seq:             {item_seqs.shape}")
            else:
                print("WARNING: --mask_history set but item_seq.npy not found; masking disabled.")

        recall_ks = [int(k.strip()) for k in args.recall_ks.split(",") if k.strip()]
        print(f"Evaluating recall@{recall_ks} ...")

        res = evaluate_precomputed(
            user_feature=user_feature,
            item_feature=item_feature,
            pos_items=pos_items,
            recall_ks=recall_ks,
            ivf_nlist=int(args.ivf_nlist),
            ivf_nprobe=int(args.ivf_nprobe),
            mask_history=bool(args.mask_history),
            item_seqs=item_seqs,
        )
        print("\n=== Precomputed ANN Recall ===")
        print(f"mask_history={args.mask_history}")
        for k in recall_ks:
            print(f"  recall@{k}: {res[f'recall@{k}']:.4f}")
        return

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
            "plm_suffix": str(args.plm_suffix),
            "plm_size": int(args.plm_size),
            "adaptor_layers": adaptor_layers,
            "train_stage": args.train_stage,
        },
    )
    init_seed(cfg["seed"], cfg["reproducibility"])
    init_logger(cfg)

    dataset = create_dataset(cfg)
    train_data, valid_data, test_data = data_preparation(cfg, dataset)
    model = model_class(cfg, train_data.dataset).to(cfg["device"])
    trainer = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, model)

    # Train with best-valid state in memory
    best_score = None
    best_state = None
    best_epoch = -1
    for epoch in range(int(cfg["epochs"])):
        trainer._train_epoch(train_data, epoch, show_progress=False)
        valid_score, valid_result = trainer._valid_epoch(valid_data, show_progress=False)
        if best_score is None or valid_score > best_score:
            best_score = float(valid_score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)

    if args.use_ann:
        res = evaluate_ann_retrieval(
            model,
            test_data,
            ann_k=int(args.ann_k),
            overfetch=int(args.ann_overfetch),
            mask_history=bool(args.mask_history),
            device=cfg["device"],
            max_eval_batches=int(args.max_eval_batches),
            ivf_nlist=int(args.ivf_nlist),
            ivf_nprobe=int(args.ivf_nprobe),
        )
    else:
        # Full-sort dot-product retrieval using model.full_sort_predict
        res = evaluate_two_tower(
            model,
            test_data,
            mask_history=bool(args.mask_history),
            device=cfg["device"],
            max_eval_batches=int(args.max_eval_batches),
        )

    print("best_epoch", best_epoch)
    print("mask_history", bool(args.mask_history))
    print("use_ann", bool(args.use_ann), "ann_k", int(args.ann_k))
    print("metrics", res)


if __name__ == "__main__":
    main()
