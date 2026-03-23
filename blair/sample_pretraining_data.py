import argparse
import json
import random

import pandas as pd
from huggingface_hub import hf_hub_download


DEFAULT_VALID_TIMESTAMP = 1628643414042


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a small (review, item-metadata) TSV for BLaIR pretraining. "
            "This version avoids `datasets.load_dataset()` dataset scripts and downloads JSONL files directly from the HF Hub."
        )
    )
    parser.add_argument(
        '--categories',
        type=str,
        default='All_Beauty',
        help="Comma-separated categories (e.g. 'All_Beauty,Books') or 'all'"
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--valid_timestamp', type=int, default=DEFAULT_VALID_TIMESTAMP)
    parser.add_argument('--downsampling_factor', type=int, default=10)
    parser.add_argument('--min_review_len', type=int, default=30)
    parser.add_argument('--min_meta_len', type=int, default=30)
    parser.add_argument('--max_pairs', type=int, default=20000, help='Stop after collecting this many pairs total')
    parser.add_argument('--output', type=str, default='clean_review_meta.tsv')
    return parser.parse_args()


def load_all_categories():
    category_filepath = hf_hub_download(
        repo_id='McAuley-Lab/Amazon-Reviews-2023',
        filename='all_categories.txt',
        repo_type='dataset'
    )
    with open(category_filepath, 'r') as file:
        all_categories = [_.strip() for _ in file.readlines()]
    return all_categories


def concat_item_metadata(dp):
    meta = ''
    flag = False
    title = dp.get('title')
    features = dp.get('features') or []
    description = dp.get('description') or []

    if title is not None:
        meta += title
        flag = True
    if len(features) > 0:
        if flag:
            meta += ' '
        meta += ' '.join(features)
        flag = True
    if len(description) > 0:
        if flag:
            meta += ' '
        meta += ' '.join(description)
    dp['cleaned_metadata'] = meta \
        .replace('\t', ' ') \
        .replace('\n', ' ') \
        .replace('\r', '') \
        .strip()
    return dp


def filter_reviews(dp, rng, downsampling_factor, valid_timestamp, all_cleaned_item_metadata, min_review_len):
    # Downsampling
    pr = rng.randint(1, downsampling_factor)
    if pr > 1:
        return False
    if dp.get('timestamp', 0) >= valid_timestamp:
        return False
    asin = dp['parent_asin']
    if asin not in all_cleaned_item_metadata:
        return False
    if len(dp.get('cleaned_review', '')) <= min_review_len:
        return False
    return True


def concat_review(dp):
    review = ''
    flag = False
    title = dp.get('title')
    text = dp.get('text')

    if title is not None:
        review += title
        flag = True
    if text is not None:
        if flag:
            review += ' '
        review += text
    dp['cleaned_review'] = review \
        .replace('\t', ' ') \
        .replace('\n', ' ') \
        .replace('\r', '') \
        .strip()
    return dp


if __name__ == '__main__':
    args = parse_args()
    rng = random.Random(args.seed)

    if args.categories.strip().lower() == 'all':
        categories = load_all_categories()
    else:
        categories = [c.strip() for c in args.categories.split(',') if c.strip()]

    all_cleaned_item_metadata = {}

    # Load item metadata (JSONL) for selected categories
    for category in categories:
        meta_filename = f'raw/meta_categories/meta_{category}.jsonl'
        meta_path = hf_hub_download(
            repo_id='McAuley-Lab/Amazon-Reviews-2023',
            filename=meta_filename,
            repo_type='dataset'
        )
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                dp = json.loads(line)
                dp = concat_item_metadata(dp)
                if len(dp.get('cleaned_metadata', '')) <= args.min_meta_len:
                    continue
                item_id = dp.get('parent_asin')
                if item_id:
                    all_cleaned_item_metadata[item_id] = dp['cleaned_metadata']

    # Load reviews (JSONL) and build (review, meta) pairs
    output_review = []
    output_metadata = []
    for category in categories:
        if len(output_review) >= args.max_pairs:
            break

        review_filename = f'raw/review_categories/{category}.jsonl'
        review_path = hf_hub_download(
            repo_id='McAuley-Lab/Amazon-Reviews-2023',
            filename=review_filename,
            repo_type='dataset'
        )
        with open(review_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(output_review) >= args.max_pairs:
                    break
                line = line.strip()
                if not line:
                    continue
                dp = json.loads(line)
                dp = concat_review(dp)
                if not filter_reviews(
                    dp,
                    rng=rng,
                    downsampling_factor=args.downsampling_factor,
                    valid_timestamp=args.valid_timestamp,
                    all_cleaned_item_metadata=all_cleaned_item_metadata,
                    min_review_len=args.min_review_len,
                ):
                    continue
                asin = dp['parent_asin']
                output_review.append(dp['cleaned_review'])
                output_metadata.append(all_cleaned_item_metadata[asin])

    df = pd.DataFrame({'review': output_review, 'meta': output_metadata})
    df.to_csv(args.output, sep='\t', lineterminator='\n', index=False)
    print(f"Saved {len(df)} pairs to {args.output}")
