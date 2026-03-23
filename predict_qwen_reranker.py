# coding=utf-8
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from llm_qwen import Qwen2ForCausalLM
from qwen_rerank_utils import (
    JsonlCandidateDataset,
    build_label_token_ids,
    compute_recall_ndcg,
    eval_collate_fn,
    gather_candidate_items,
    load_item_titles,
    rank_candidate_scores,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the best-valid Qwen reranker on test candidates.")
    parser.add_argument("--test_candidates", type=str, default="artifacts/qwen_rerank/test.top50.jsonl")
    parser.add_argument("--meta_jsonl", type=str, default="data/raw/meta_categories/meta_All_Beauty.jsonl")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint_dir", type=str, default="artifacts/qwen_rerank/qwen_model/best_model")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_candidates", type=int, default=50)
    parser.add_argument("--max_title_len", type=int, default=32)
    parser.add_argument("--max_text_len", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_jsonl", type=str, default="artifacts/qwen_rerank/test_predictions.jsonl")
    parser.add_argument("--candidate_summary_json", type=str, default="artifacts/qwen_rerank/candidate_summary.json")
    parser.add_argument(
        "--test_inter_path",
        type=str,
        default="seq_rec_results/dataset/processed/All_Beauty/All_Beauty.test.inter",
    )
    return parser.parse_args()


def count_interactions(inter_path: str) -> int:
    total = 0
    with open(inter_path, "r", encoding="utf-8") as fin:
        _ = fin.readline()
        for line in fin:
            if line.strip():
                total += 1
    return total


def load_total_examples(args, included_count: int) -> int:
    summary_path = Path(args.candidate_summary_json)
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as fin:
            summary = json.load(fin)
        split_summary = summary.get("export_summary", {}).get("test", {})
        rows = int(split_summary.get("rows", included_count))
        skipped = int(split_summary.get("skipped_not_in_top1000", 0))
        return rows + skipped
    inter_path = Path(args.test_inter_path)
    if inter_path.exists():
        return count_interactions(str(inter_path))
    return included_count


def load_model(args):
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}. "
            "Run `train_qwen_reranker.py` successfully first, or pass a valid local checkpoint path."
        )
    if (checkpoint_dir / "adapter_config.json").exists():
        from peft import PeftModel

        base_model = Qwen2ForCausalLM.from_pretrained(args.base_model, cache_dir=args.cache_dir)
        model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
    else:
        config_path = checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Missing `config.json` under {checkpoint_dir}. "
                "This usually means training did not finish saving the model."
            )
        model = Qwen2ForCausalLM.from_pretrained(str(checkpoint_dir), cache_dir=args.cache_dir)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model


@torch.no_grad()
def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    need_items = gather_candidate_items([args.test_candidates])
    title_map = load_item_titles(args.meta_jsonl, need_items=need_items)
    label_token_ids = build_label_token_ids(tokenizer, args.num_candidates)

    dataset = JsonlCandidateDataset(
        path=args.test_candidates,
        tokenizer=tokenizer,
        title_map=title_map,
        max_title_len=args.max_title_len,
        max_text_len=args.max_text_len,
        num_candidates=args.num_candidates,
        train=False,
        train_on_inputs=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: eval_collate_fn(batch, tokenizer.pad_token_id, include_metadata=True),
    )

    model = load_model(args)
    device = next(model.parameters()).device

    all_scores = []
    all_labels = []
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            vocab_logits = outputs.logits
            cand_scores = vocab_logits[:, label_token_ids].detach().cpu()
            sorted_scores, sorted_indices = rank_candidate_scores(cand_scores)
            labels = batch["labels"]
            all_scores.append(cand_scores)
            all_labels.append(labels)

            for i in range(cand_scores.size(0)):
                ranked_candidates = [batch["candidate_tokens"][i][j] for j in sorted_indices[i].tolist()]
                payload = {
                    "user_token": batch["user_token"][i],
                    "history_tokens": batch["history_tokens"][i],
                    "pos_item_token": batch["pos_item_token"][i],
                    "candidate_tokens": batch["candidate_tokens"][i],
                    "candidate_scores": cand_scores[i].tolist(),
                    "ranked_candidate_tokens": ranked_candidates,
                    "ranked_candidate_scores": sorted_scores[i].tolist(),
                }
                fout.write(json.dumps(payload) + "\n")

    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    recall10, ndcg10 = compute_recall_ndcg(all_scores, all_labels, k=10)
    included_count = int(all_labels.size(0))
    total_count = load_total_examples(args, included_count)
    coverage = included_count / max(total_count, 1)
    summary = {
        "conditional": {
            "num_examples": included_count,
            "recall@10": recall10,
            "ndcg@10": ndcg10,
        },
        "end_to_end": {
            "num_examples": total_count,
            "coverage": coverage,
            "recall@10": recall10 * coverage,
            "ndcg@10": ndcg10 * coverage,
        },
        "prediction_file": str(output_path),
    }
    with open(output_path.with_suffix(".metrics.json"), "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
