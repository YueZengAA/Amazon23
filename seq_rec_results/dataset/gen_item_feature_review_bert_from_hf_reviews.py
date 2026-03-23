import argparse
import gzip
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from transformers import AutoModel, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Download Amazon Reviews 2023 raw review JSONL.GZ for a domain, encode each review title+text "
            "with BERT CLS last hidden state, average per item, and write a RecBole .feature file."
        )
    )
    p.add_argument("--domain", type=str, default="All_Beauty")
    p.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("seq_rec_results", "dataset", "processed"),
        help="RecBole processed root; expects <data_dir>/<domain>/<domain>.data_maps",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.join("data", "amazon_reviews_2023", ".cache", "huggingface", "download"),
        help="Directory used to cache the downloaded raw review .jsonl.gz file.",
    )
    p.add_argument(
        "--review_url",
        type=str,
        default="",
        help="Optional direct URL to review JSONL.GZ. Defaults to the UCSD data repo URL for the domain.",
    )
    p.add_argument("--model", type=str, default="bert-base-uncased")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--output_name",
        type=str,
        default="All_Beauty.review_bert.feature",
        help="Output feature filename under <data_dir>/<domain>/.",
    )
    return p.parse_args()


def clean_text(s):
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return " ".join(s.split()).strip()


def review_to_text(title, text):
    title = clean_text(title)
    text = clean_text(text)
    if title and text:
        return f"{title} {text}"
    return title or text


def load_data_maps(maps_path):
    with open(maps_path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_review_url(domain):
    return (
        "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
        f"raw/review_categories/{domain}.jsonl?download=true"
    )
 

def ensure_review_file(domain, cache_dir, review_url, timeout):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_jsonl = cache_dir / f"raw_review_{domain}.jsonl"
    base_gz = cache_dir / f"raw_review_{domain}.jsonl.gz"
    for cached in (base_jsonl, base_gz):
        if cached.exists() and cached.stat().st_size > 0:
            print(f"Using cached review file: {cached}")
            return cached

    url = review_url or default_review_url(domain)
    parsed_path = urlparse(url).path.lower()
    out_path = base_gz if parsed_path.endswith(".gz") else base_jsonl
    print(f"Downloading review data from: {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "wb") as fout:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fout.write(chunk)
        os.replace(tmp_path, out_path)
    print(f"Saved review file: {out_path}")
    return out_path


def encode_cls(model, tokenizer, texts, device, max_length):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        feat = out.last_hidden_state[:, 0, :]
    return feat.detach().cpu().numpy().astype(np.float32)


def batched(iterable, batch_size):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def iter_review_records(gz_path):
    opener = gzip.open if str(gz_path).lower().endswith(".gz") else open
    with opener(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    args = parse_args()

    data_path = Path(args.data_dir) / args.domain
    maps_path = data_path / f"{args.domain}.data_maps"
    if not maps_path.exists():
        raise FileNotFoundError(f"Missing data_maps: {maps_path}")
    data_path.mkdir(parents=True, exist_ok=True)

    out_path = data_path / args.output_name
    maps = load_data_maps(str(maps_path))
    ordered_items = maps["id2item"][1:]
    item2row = {asin: idx for idx, asin in enumerate(ordered_items)}

    gz_path = ensure_review_file(
        domain=args.domain,
        cache_dir=args.cache_dir,
        review_url=args.review_url,
        timeout=int(args.timeout),
    )

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval().to(device)

    hidden_size = int(getattr(model.config, "hidden_size", 768))
    sums = np.zeros((len(ordered_items), hidden_size), dtype=np.float64)
    counts = np.zeros(len(ordered_items), dtype=np.int64)

    kept_reviews = 0
    processed = 0
    batch_size = int(args.batch_size)

    def review_examples():
        for dp in iter_review_records(gz_path):
            asin = dp.get("parent_asin")
            row = item2row.get(asin)
            if row is None:
                continue
            text = review_to_text(dp.get("title"), dp.get("text"))
            if not text:
                continue
            yield row, text

    for batch in batched(review_examples(), batch_size):
        rows = [row for row, _ in batch]
        texts = [text for _, text in batch]
        feat = encode_cls(model, tokenizer, texts, device, int(args.max_length))
        np.add.at(sums, np.asarray(rows, dtype=np.int64), feat.astype(np.float64))
        np.add.at(counts, np.asarray(rows, dtype=np.int64), 1)
        kept_reviews += len(batch)
        processed += len(batch)
        if (processed // batch_size) % 200 == 0:
            covered = int((counts > 0).sum())
            print(f"- encoded {processed} reviews, covered_items={covered}")

    features = np.zeros((len(ordered_items), hidden_size), dtype=np.float32)
    nonzero = counts > 0
    if np.any(nonzero):
        features[nonzero] = (sums[nonzero] / counts[nonzero, None]).astype(np.float32)

    tmp_path = str(out_path) + ".tmp"
    with open(tmp_path, "wb") as fout:
        features.tofile(fout)
    os.replace(tmp_path, out_path)

    print(f"Wrote: {out_path}")
    print(f"- shape: ({features.shape[0]}, {features.shape[1]}) float32")
    print(f"- items with >=1 review: {int(nonzero.sum())}/{len(ordered_items)}")
    print(f"- reviews encoded: {kept_reviews}")


if __name__ == "__main__":
    main()
