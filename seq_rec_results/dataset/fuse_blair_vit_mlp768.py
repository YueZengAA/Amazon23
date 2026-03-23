import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fuse BLaIR(768) + ViT(768) item embeddings by concatenation -> MLP -> 768, "
            "and write a new RecBole .feature file aligned with <domain>.data_maps id2item order."
        )
    )
    p.add_argument("--domain", type=str, default="All_Beauty")
    p.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("seq_rec_results", "dataset", "processed"),
        help="RecBole processed root; expects <data_dir>/<domain>/<domain>.data_maps",
    )
    p.add_argument("--blair_suffix", type=str, default="blair768.feature")
    p.add_argument("--vit_suffix", type=str, default="vit_img768.feature")
    p.add_argument("--output_suffix", type=str, default="blair_vit_mlp768.feature")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse"],
        help="Training objective (self-supervised).",
    )
    p.add_argument(
        "--w_blair",
        type=float,
        default=1.0,
        help="Weight for reconstructing the BLaIR half (first 768 dims).",
    )
    p.add_argument(
        "--w_vit",
        type=float,
        default=1.0,
        help="Weight for reconstructing the ViT half (last 768 dims).",
    )

    p.add_argument(
        "--init",
        type=str,
        default="keep_blair",
        choices=["keep_blair", "random"],
        help=(
            "Initialization for encoder projection. keep_blair initializes the 1536->768 layer so that "
            "output starts as BLaIR (ignoring ViT)."
        ),
    )
    p.add_argument("--normalize_output", action="store_true", help="L2-normalize final 768-d vectors.")
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_maps(maps_path: str) -> dict:
    with open(maps_path, "r", encoding="utf-8") as f:
        return json.load(f)


def expected_rows(maps: dict) -> int:
    # id2item includes PAD at index 0
    return len(maps["id2item"]) - 1


def open_feature_memmap(path: str, rows: int, dim: int) -> np.memmap:
    exp_bytes = rows * dim * 4
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    size = os.path.getsize(path)
    if size != exp_bytes:
        raise RuntimeError(f"Unexpected feature file size: {path} size={size} expected={exp_bytes}")
    return np.memmap(path, dtype=np.float32, mode="r", shape=(rows, dim))


@dataclass
class Stats:
    rows: int
    dim_blair: int
    dim_vit: int


class FuseAE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float, init: str):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden, out_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(out_dim, hidden),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden, in_dim),
        )

        if init == "keep_blair":
            # Make encoder output ~= first out_dim dims of input at init.
            # This uses the last Linear(hidden->out_dim) layer, so we also zero-init the first layer.
            # Practically, we set both linears so the whole enc approximates [I, 0] projection.
            l0: nn.Linear = self.enc[0]  # type: ignore
            l3: nn.Linear = self.enc[3]  # type: ignore
            nn.init.zeros_(l0.weight)
            nn.init.zeros_(l0.bias)
            nn.init.zeros_(l3.weight)
            nn.init.zeros_(l3.bias)
        elif init == "random":
            pass
        else:
            raise ValueError(init)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.enc(x)
        recon = self.dec(z)
        return z, recon


def main():
    args = parse_args()
    set_seed(int(args.seed))

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = Path(str(args.data_dir)).expanduser()
    if not data_dir.is_absolute():
        data_dir = (repo_root / data_dir).resolve()

    domain = args.domain
    data_path = data_dir / domain
    maps_path = data_path / f"{domain}.data_maps"
    maps = load_maps(maps_path)
    rows = expected_rows(maps)

    dim_blair = 768
    dim_vit = 768
    in_dim = dim_blair + dim_vit
    out_dim = 768

    blair_path = data_path / f"{domain}.{args.blair_suffix}"
    vit_path = data_path / f"{domain}.{args.vit_suffix}"
    out_path = data_path / f"{domain}.{args.output_suffix}"
    tmp_path = str(out_path) + ".tmp"

    blair = open_feature_memmap(str(blair_path), rows=rows, dim=dim_blair)
    vit = open_feature_memmap(str(vit_path), rows=rows, dim=dim_vit)

    device = torch.device(args.device)
    model = FuseAE(in_dim=in_dim, hidden=int(args.hidden), out_dim=out_dim, dropout=float(args.dropout), init=str(args.init))
    model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    bs = int(args.batch_size)
    n = rows
    order = np.arange(n)

    w_blair = float(args.w_blair)
    w_vit = float(args.w_vit)

    for epoch in range(int(args.epochs)):
        np.random.shuffle(order)
        total = 0.0
        steps = 0
        for s in range(0, n, bs):
            idx = order[s : s + bs]
            xb = np.concatenate([blair[idx], vit[idx]], axis=1)
            x = torch.from_numpy(xb).to(device)

            z, recon = model(x)
            if args.loss == "mse":
                # weighted mse per half
                t = x
                r = recon
                loss_bl = ((r[:, :dim_blair] - t[:, :dim_blair]) ** 2).mean()
                loss_v = ((r[:, dim_blair:] - t[:, dim_blair:]) ** 2).mean()
                loss = w_blair * loss_bl + w_vit * loss_v
            else:
                raise ValueError(args.loss)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += float(loss.item())
            steps += 1
        print(f"epoch {epoch+1}/{int(args.epochs)} loss={total/max(1,steps):.6f}")

    # Export fused 768d feature
    model.eval()
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    with torch.no_grad(), open(tmp_path, "wb") as fout:
        for s in range(0, n, bs):
            idx = slice(s, min(n, s + bs))
            xb = np.concatenate([blair[idx], vit[idx]], axis=1)
            x = torch.from_numpy(np.asarray(xb, dtype=np.float32)).to(device)
            z, _ = model(x)
            if bool(args.normalize_output):
                z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            z.detach().cpu().numpy().astype(np.float32).tofile(fout)
            if (s // bs) % 200 == 0:
                print(f"- wrote {min(n, s+bs)}/{n}")

    try:
        os.replace(tmp_path, str(out_path))
    except Exception:
        os.rename(tmp_path, str(out_path))

    print("Wrote", str(out_path))
    print("- shape", (rows, out_dim), "float32")


if __name__ == "__main__":
    main()
