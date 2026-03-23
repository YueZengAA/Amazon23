import argparse
import os
import json

import numpy as np
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generate ViT image features for items from a local image cache directory. "
            "Writes a RecBole .feature binary (float32) aligned to <domain>.data_maps id2item order."
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
        "--image_dir",
        type=str,
        default=os.path.join("data", "amazon_reviews_2023", "image_cache", "All_Beauty_256"),
        help="Directory containing 256x256 JPEGs named by parent_asin (safe_filename in downloader).",
    )
    p.add_argument("--model", type=str, default="google/vit-base-patch16-224-in21k")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument(
        "--pooler",
        type=str,
        choices=["cls", "mean"],
        default="cls",
        help="Feature pooling on ViT last_hidden_state.",
    )
    p.add_argument(
        "--output_suffix",
        type=str,
        default="vit_img768.feature",
        help="Output file suffix (written under <data_dir>/<domain>/<domain>.<suffix>).",
    )
    return p.parse_args()


def safe_filename(s: str) -> str:
    return "".join([c if c.isalnum() else "_" for c in s])[:200]


def load_data_maps(maps_path: str) -> dict:
    with open(maps_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image(image_dir: str, asin: str) -> Image.Image | None:
    p = os.path.join(image_dir, safe_filename(asin) + ".jpg")
    if not os.path.exists(p):
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


@torch.no_grad()
def encode_batch(model: ViTModel, processor: ViTImageProcessor, images: list[Image.Image], device: str, pooler: str) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    hs = out.last_hidden_state  # [B, 1+N, H]
    if pooler == "cls":
        feat = hs[:, 0]
    elif pooler == "mean":
        feat = hs.mean(dim=1)
    else:
        raise ValueError(pooler)
    feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return feat.detach().cpu().numpy().astype(np.float32)


def main():
    args = parse_args()
    domain = args.domain
    data_path = os.path.join(args.data_dir, domain)
    maps_path = os.path.join(data_path, f"{domain}.data_maps")
    if not os.path.exists(maps_path):
        raise FileNotFoundError(f"Missing data_maps: {maps_path}. Build RecBole dataset first.")
    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"Missing image_dir: {args.image_dir}")

    out_path = os.path.join(data_path, f"{domain}.{args.output_suffix}")
    tmp_path = out_path + ".tmp"

    maps = load_data_maps(maps_path)
    ordered_items = maps["id2item"][1:]  # exclude PAD

    processor = ViTImageProcessor.from_pretrained(args.model)
    model = ViTModel.from_pretrained(args.model)
    model.eval().to(args.device)

    hidden = int(getattr(model.config, "hidden_size", 768))
    expected_rows = len(ordered_items)
    expected_bytes = expected_rows * hidden * 4

    if os.path.exists(out_path) and os.path.getsize(out_path) == expected_bytes:
        print(f"Exists: {out_path}")
        print(f"- shape: ({expected_rows}, {hidden}) float32")
        return

    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    bs = int(args.batch_size)
    missing = 0
    written = 0

    with open(tmp_path, "wb") as fout:
        i = 0
        while i < len(ordered_items):
            batch_items = ordered_items[i : i + bs]
            images: list[Image.Image] = []
            idxs: list[int] = []
            for j, asin in enumerate(batch_items):
                img = load_image(args.image_dir, asin)
                if img is None:
                    missing += 1
                    continue
                images.append(img)
                idxs.append(j)

            feats = np.zeros((len(batch_items), hidden), dtype=np.float32)
            if images:
                enc = encode_batch(model, processor, images, device=args.device, pooler=str(args.pooler))
                for j, f in zip(idxs, enc):
                    feats[j] = f

            feats.tofile(fout)
            i += len(batch_items)
            written = i
            if (i // bs) % 200 == 0:
                print(f"- wrote {i}/{len(ordered_items)} missing_images={missing}")

    try:
        os.replace(tmp_path, out_path)
    except Exception:
        os.rename(tmp_path, out_path)

    final_size = os.path.getsize(out_path)
    if final_size != expected_bytes:
        print(
            f"WARNING: wrote {out_path} but size {final_size} bytes != expected {expected_bytes}. File may be incomplete."
        )
    print(f"Wrote: {out_path}")
    print(f"- shape: ({expected_rows}, {hidden}) float32")
    print(f"- missing_images: {missing}")


if __name__ == "__main__":
    main()
