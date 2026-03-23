import argparse
import json
import os
from typing import Iterable

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build RecBole sequential .inter files + data_maps from local timestamp_w_his CSVs "
            "(Amazon Reviews 2023 benchmark format)."
        )
    )
    p.add_argument(
        "--domain",
        type=str,
        default="All_Beauty",
        help="Domain name, e.g. All_Beauty",
    )
    p.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join("data", "amazon_reviews_2023", "benchmark", "0core", "timestamp_w_his"),
        help="Directory containing <domain>.{train,valid,test}.csv",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("seq_rec_results", "dataset", "processed"),
        help="RecBole data_path root; will create <output_dir>/<domain>/",
    )
    p.add_argument(
        "--max_his_len",
        type=int,
        default=50,
        help="Truncate history to this length (keep most recent)",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="CSV read chunksize (reduce if RAM is limited)",
    )
    return p.parse_args()


def iter_csv(path: str, chunksize: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(path, chunksize=chunksize)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_history(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    return s


def truncate_history(his: str, max_len: int) -> str:
    if not his:
        return ""
    xs = his.split()
    if len(xs) <= max_len:
        return " ".join(xs)
    return " ".join(xs[-max_len:])


def update_sets(df: pd.DataFrame, users: set, items: set) -> None:
    if "user_id" in df.columns:
        users.update(df["user_id"].astype(str).tolist())
    if "parent_asin" in df.columns:
        items.update(df["parent_asin"].astype(str).tolist())
    if "history" in df.columns:
        for his in df["history"].dropna().astype(str).tolist():
            his = his.strip()
            if not his:
                continue
            items.update(his.split())


def write_inter_file(split: str, in_path: str, out_path: str, max_his_len: int, chunksize: int) -> None:
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("user_id:token\titem_id_list:token_seq\titem_id:token\n")
        for chunk in iter_csv(in_path, chunksize):
            if "history" not in chunk.columns:
                raise ValueError(f"Missing 'history' column in {in_path}")
            chunk = chunk[chunk["history"].notna()].copy()
            chunk["history"] = chunk["history"].astype(str).str.strip()
            chunk = chunk[chunk["history"] != ""]
            if chunk.empty:
                continue
            users = chunk["user_id"].astype(str).tolist()
            targets = chunk["parent_asin"].astype(str).tolist()
            histories = chunk["history"].astype(str).tolist()
            for u, t, his in zip(users, targets, histories):
                his2 = truncate_history(clean_history(his), max_his_len)
                if not his2:
                    continue
                f.write(f"{u}\t{his2}\t{t}\n")


def main():
    args = parse_args()

    domain = args.domain
    in_dir = args.input_dir
    out_root = args.output_dir
    out_dir = os.path.join(out_root, domain)
    ensure_dir(out_dir)

    splits = ["train", "valid", "test"]
    in_paths = {s: os.path.join(in_dir, f"{domain}.{s}.csv") for s in splits}
    for s, pth in in_paths.items():
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Missing input: {pth}")

    print(f"Building dataset {domain}")
    print(f"- input_dir: {in_dir}")
    print(f"- output_dir: {out_dir}")

    users = set()
    items = set()
    for s in splits:
        for chunk in iter_csv(in_paths[s], args.chunksize):
            update_sets(chunk, users, items)
        print(f"- scanned {s}")

    user2id = {"[PAD]": 0}
    id2user = ["[PAD]"]
    for u in sorted(users):
        user2id[u] = len(id2user)
        id2user.append(u)

    item2id = {"[PAD]": 0}
    id2item = ["[PAD]"]
    for it in sorted(items):
        item2id[it] = len(id2item)
        id2item.append(it)

    data_maps = {
        "user2id": user2id,
        "id2user": id2user,
        "item2id": item2id,
        "id2item": id2item,
    }
    maps_path = os.path.join(out_dir, f"{domain}.data_maps")
    with open(maps_path, "w", encoding="utf-8") as f:
        json.dump(data_maps, f)
    print(f"- wrote {maps_path}")
    print(f"- users: {len(id2user) - 1}")
    print(f"- items: {len(id2item) - 1}")

    for s in splits:
        out_path = os.path.join(out_dir, f"{domain}.{s}.inter")
        write_inter_file(s, in_paths[s], out_path, args.max_his_len, args.chunksize)
        print(f"- wrote {out_path}")


if __name__ == "__main__":
    main()
