import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate BLaIR item feature file from meta JSONL (raw/meta_categories).")
    p.add_argument("--domain", type=str, default="All_Beauty")
    p.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("seq_rec_results", "dataset", "processed"),
        help="RecBole data_path root; expects <data_dir>/<domain>/<domain>.data_maps",
    )
    p.add_argument(
        "--meta_jsonl",
        type=str,
        default=os.path.join("data", "amazon_reviews_2023", "raw", "meta_categories", "meta_All_Beauty.jsonl"),
        help="Path to meta_<domain>.jsonl",
    )
    p.add_argument("--model", type=str, default="hyp1231/blair-roberta-base")
    p.add_argument("--batch_size", type=int, default=64)
    # Paper uses RoBERTa tokenizer and truncates to 64 tokens.
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument(
        "--pooler",
        type=str,
        choices=["cls", "cls_before_pooler", "mean"],
        default="cls",
        help="Pooling for sentence embedding. 'cls' uses outputs.pooler_output (SimCSE supervised default).",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_suffix", type=str, default="blair768.feature")
    return p.parse_args()


def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return " ".join(s.split()).strip()


def meta_to_text(dp: dict) -> str:
    title = clean_text(dp.get("title"))
    features = dp.get("features") or []
    description = dp.get("description") or []

    parts = []
    if title:
        parts.append(title)
    if isinstance(features, list) and features:
        parts.append(clean_text(" ".join([clean_text(x) for x in features if x is not None])))
    if isinstance(description, list) and description:
        parts.append(clean_text(" ".join([clean_text(x) for x in description if x is not None])))
    return clean_text(" ".join(parts))


def load_data_maps(maps_path: str) -> dict:
    with open(maps_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_item_texts(meta_jsonl: str, need_items: set) -> dict:
    item2text = {}
    with open(meta_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                dp = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = dp.get("parent_asin")
            if not asin or asin not in need_items:
                continue
            if asin in item2text:
                continue
            item2text[asin] = meta_to_text(dp)
    return item2text


def encode(model, tok, texts, device, max_length, pooler: str):
    inputs = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        if pooler == "cls":
            pooled = out.pooler_output
        elif pooler == "cls_before_pooler":
            pooled = out.last_hidden_state[:, 0]
        elif pooler == "mean":
            last = out.last_hidden_state
            mask = inputs.get("attention_mask")
            if mask is None:
                pooled = last.mean(dim=1)
            else:
                m = mask.unsqueeze(-1).type_as(last)
                denom = m.sum(dim=1).clamp(min=1.0)
                pooled = (last * m).sum(dim=1) / denom
        else:
            raise ValueError(f"Unknown pooler: {pooler}")
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
    return pooled.detach().cpu().numpy().astype(np.float32)


def main():
    args = parse_args()
    domain = args.domain
    data_path = os.path.join(args.data_dir, domain)
    maps_path = os.path.join(data_path, f"{domain}.data_maps")
    if not os.path.exists(maps_path):
        raise FileNotFoundError(f"Missing data_maps: {maps_path}. Run build_from_timestamp_w_his_csv.py first.")
    if not os.path.exists(args.meta_jsonl):
        raise FileNotFoundError(f"Missing meta_jsonl: {args.meta_jsonl}")

    out_path = os.path.join(data_path, f"{domain}.{args.output_suffix}")
    if os.path.exists(out_path):
        print(f"Exists: {out_path}")
        return

    maps = load_data_maps(maps_path)
    ordered_items = maps["id2item"][1:]  # exclude PAD
    need_items = set(ordered_items)
    print(f"Loading meta texts for {len(need_items)} items...")
    item2text = load_item_texts(args.meta_jsonl, need_items)
    print(f"- found texts: {len(item2text)}")

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval().to(device)

    dim = int(getattr(model.config, "hidden_size", 768))
    if dim != 768:
        print(f"WARNING: hidden_size={dim} (expected 768 for roberta-base).")

    bs = args.batch_size
    with open(out_path, "wb") as fout:
        for i in range(0, len(ordered_items), bs):
            batch_items = ordered_items[i:i + bs]
            texts = [item2text.get(it, "") for it in batch_items]
            # if all empty, write zeros
            if all((not t) for t in texts):
                z = np.zeros((len(texts), dim), dtype=np.float32)
                z.tofile(fout)
                continue
            feat = encode(model, tok, texts, device, args.max_length, args.pooler)
            feat.tofile(fout)
            if (i // bs) % 200 == 0:
                print(f"- encoded {i}/{len(ordered_items)}")

    print(f"Wrote: {out_path}")
    print(f"- shape: ({len(ordered_items)}, {dim}) float32")


if __name__ == "__main__":
    main()
