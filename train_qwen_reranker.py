# coding=utf-8
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

from llm_qwen import Qwen2ForCausalLM
from qwen_rerank_utils import (
    JsonlCandidateDataset,
    build_label_token_ids,
    eval_collate_fn,
    gather_candidate_items,
    load_item_titles,
    summarize_metrics,
    train_collate_fn,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Qwen reranker on top-50 UniSRec candidates.")
    parser.add_argument("--train_candidates", type=str, default="artifacts/qwen_rerank/train.top50.jsonl")
    parser.add_argument("--valid_candidates", type=str, default="artifacts/qwen_rerank/valid.top50.jsonl")
    parser.add_argument("--meta_jsonl", type=str, default="data/raw/meta_categories/meta_All_Beauty.jsonl")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="artifacts/qwen_rerank/qwen_model")
    parser.add_argument("--num_candidates", type=int, default=50)
    parser.add_argument("--max_title_len", type=int, default=32)
    parser.add_argument("--max_text_len", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--train_on_inputs", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(args):
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = Qwen2ForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        device_map="auto" if args.load_in_4bit else None,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )

    if args.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.config.use_cache = False
    return model


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    need_items = gather_candidate_items([args.train_candidates, args.valid_candidates])
    title_map = load_item_titles(args.meta_jsonl, need_items=need_items)
    label_token_ids = build_label_token_ids(tokenizer, args.num_candidates)

    train_dataset = JsonlCandidateDataset(
        path=args.train_candidates,
        tokenizer=tokenizer,
        title_map=title_map,
        max_title_len=args.max_title_len,
        max_text_len=args.max_text_len,
        num_candidates=args.num_candidates,
        train=True,
        train_on_inputs=args.train_on_inputs,
    )
    valid_dataset = JsonlCandidateDataset(
        path=args.valid_candidates,
        tokenizer=tokenizer,
        title_map=title_map,
        max_title_len=args.max_title_len,
        max_text_len=args.max_text_len,
        num_candidates=args.num_candidates,
        train=False,
        train_on_inputs=args.train_on_inputs,
    )

    model = build_model(args)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits[:, label_token_ids]

    def compute_metrics(eval_pred):
        scores, labels = eval_pred
        scores = torch.tensor(scores)
        labels = torch.tensor(labels).view(-1)
        return summarize_metrics(scores, labels)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="ndcg_at_10",
        greater_is_better=True,
        bf16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    class EvalTrainer(Trainer):
        def get_eval_dataloader(self, eval_dataset=None):
            self.data_collator = lambda batch: eval_collate_fn(batch, tokenizer.pad_token_id, include_metadata=False)
            return super().get_eval_dataloader(eval_dataset)

        def get_train_dataloader(self):
            self.data_collator = lambda batch: train_collate_fn(batch, tokenizer.pad_token_id)
            return super().get_train_dataloader()

    trainer = EvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=lambda batch: train_collate_fn(batch, tokenizer.pad_token_id),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    metadata = {
        "base_model": args.base_model,
        "label_token_ids": label_token_ids,
        "num_candidates": args.num_candidates,
        "max_title_len": args.max_title_len,
        "max_text_len": args.max_text_len,
        "train_candidates": args.train_candidates,
        "valid_candidates": args.valid_candidates,
        "train_result": train_result.metrics,
        "best_metric": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }
    with open(output_dir / "train_metadata.json", "w", encoding="utf-8") as fout:
        json.dump(metadata, fout, indent=2)
    print("wrote", output_dir / "train_metadata.json")


if __name__ == "__main__":
    main()
