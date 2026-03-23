import argparse
import csv
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple

import requests
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Download one image per item (from meta_<domain>.jsonl) and save as resized 256x256 JPEG. "
            "Skips items whose output file already exists."
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
        "--meta_jsonl",
        type=str,
        default=os.path.join("data", "amazon_reviews_2023", "raw", "meta_categories", "meta_All_Beauty.jsonl"),
        help="Path to meta_<domain>.jsonl",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("data", "amazon_reviews_2023", "image_cache", "All_Beauty_256"),
        help="Output directory for resized JPEGs",
    )
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--quality", type=int, default=85)
    p.add_argument("--timeout", type=int, default=15)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max_items", type=int, default=0, help="0 means all")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep seconds after each successful download (rate limiting).",
    )
    return p.parse_args()


def safe_filename(s: str) -> str:
    return "".join([c if c.isalnum() else "_" for c in s])[:200]


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    return img.crop((left, top, left + m, top + m))


def resize_256(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    img = center_crop_square(img)
    return img.resize((size, size), resample=Image.BICUBIC)


def pick_image_url(dp: dict) -> Optional[str]:
    imgs = dp.get("images")
    if not imgs:
        return None
    if isinstance(imgs, list) and imgs and isinstance(imgs[0], dict):
        for key in ("hi_res", "large", "thumb"):
            u = imgs[0].get(key)
            if isinstance(u, str) and u.startswith("http"):
                return u
        for v in imgs[0].values():
            if isinstance(v, str) and v.startswith("http"):
                return v
    return None


def load_data_maps(maps_path: str) -> dict:
    with open(maps_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_item2url(meta_jsonl: str, need_items: set) -> dict:
    item2url = {}
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
            if asin in item2url:
                continue
            item2url[asin] = pick_image_url(dp)
    return item2url


@dataclass
class Result:
    asin: str
    url: str
    out_path: str
    ok: bool
    err: str


def download_and_save(session: requests.Session, asin: str, url: str, out_path: str, size: int, quality: int, timeout: int) -> Result:
    if not url:
        return Result(asin=asin, url="", out_path=out_path, ok=False, err="no_url")
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        img = resize_256(img, size=size)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp"
        img.save(tmp, format="JPEG", quality=int(quality), optimize=True)
        os.replace(tmp, out_path)
        return Result(asin=asin, url=url, out_path=out_path, ok=True, err="")
    except Exception as e:
        return Result(asin=asin, url=url, out_path=out_path, ok=False, err=repr(e))


def main():
    args = parse_args()
    domain = args.domain
    data_path = os.path.join(args.data_dir, domain)
    maps_path = os.path.join(data_path, f"{domain}.data_maps")
    if not os.path.exists(maps_path):
        raise FileNotFoundError(f"Missing data_maps: {maps_path}. Build RecBole dataset first.")
    if not os.path.exists(args.meta_jsonl):
        raise FileNotFoundError(f"Missing meta_jsonl: {args.meta_jsonl}")

    maps = load_data_maps(maps_path)
    ordered_items = maps["id2item"][1:]  # exclude PAD
    if int(args.max_items) and int(args.max_items) > 0:
        ordered_items = ordered_items[: int(args.max_items)]

    need_items = set(ordered_items)
    print("items", len(ordered_items))
    print("loading_meta_urls...")
    item2url = build_item2url(args.meta_jsonl, need_items)
    urls_found = sum(1 for v in item2url.values() if v)
    print("urls_found", urls_found)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, f"{domain}.download_manifest.csv")

    # Build tasks
    tasks: list[Tuple[str, str, str]] = []
    skipped = 0
    for asin in ordered_items:
        out_path = os.path.join(out_dir, safe_filename(asin) + ".jpg")
        if (not args.overwrite) and os.path.exists(out_path):
            skipped += 1
            continue
        url = item2url.get(asin) or ""
        tasks.append((asin, url, out_path))
    print("already_exists", skipped)
    print("to_download", len(tasks))

    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; AmazonReviews2023ImageDownloader/1.0)"})

    ok = 0
    fail = 0
    t0 = time.time()

    # Stream results to manifest
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["asin", "url", "out_path", "ok", "err"])

        with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = []
            for asin, url, out_path in tasks:
                futs.append(
                    ex.submit(
                        download_and_save,
                        sess,
                        asin,
                        url,
                        out_path,
                        int(args.size),
                        int(args.quality),
                        int(args.timeout),
                    )
                )

            for i, fut in enumerate(as_completed(futs), 1):
                r: Result = fut.result()
                w.writerow([r.asin, r.url, r.out_path, int(r.ok), r.err])
                if r.ok:
                    ok += 1
                    if args.sleep and float(args.sleep) > 0:
                        time.sleep(float(args.sleep))
                else:
                    fail += 1
                if i % 200 == 0:
                    dt = max(1e-6, time.time() - t0)
                    print(f"done {i}/{len(futs)} ok={ok} fail={fail} rate={i/dt:.2f}/s")

    dt = max(1e-6, time.time() - t0)
    print("finished")
    print("ok", ok, "fail", fail, "skipped", skipped)
    print("seconds", round(dt, 2))
    print("manifest", manifest_path)


if __name__ == "__main__":
    main()
