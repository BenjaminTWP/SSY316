#!/usr/bin/env python3
"""
P3 (Mini project) — Code template 3/3: VAE + latent diffusion for MNIST (P3.3)
Based on the exam instructions. fileciteturn14file0

This skeleton covers:
- P3.3.2 encode MNIST -> latent dataset {z_i} (using z = mu(x))
- P3.3.3 learn latent score with Hyvärinen + Hutchinson divergence
- P3.3.4 reverse sampling in latent space + decode to images
- P3.3.5 compare VAE-only vs latent diffusion

Students must fill:
- Hutchinson divergence helper
- (optionally) better sampling / plots / hyperparams
- (optional Q4) conditional extension s(z,t,y)

Run:
  python P3_P3.3_latent_diffusion_template.py
"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from P3_1_vae_template import VAE  # Reuse VAE from Q1
from P3_2_spiral_score_template import (
    ScoreNet,
    divergence_hutchinson,
    reverse_euler,
    alpha_bar,
)


@dataclass
class LatentConfig:
    vae_weights: str = (
        "outputs_q1_vae/vae_mnist.pt"  # TODO: set path to your trained VAE
    )
    latent_dim: int = 2  # TODO: choose  # MUST match your VAE dz
    hidden_dim: int = 256  # TODO: choose  # MUST match your VAE hidden

    score_hidden: int = 256  # TODO: choose

    beta0: float = 0
    beta1: float = 0.2
    K: int = 200

    steps: int = 6000
    batch_size: int = 1024
    lr: float = 2e-4

    outdir: str = "outputs_P3.3_latent"
    seed: int = 0


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_noise(z0, k, ab, cfg: LatentConfig) -> np.ndarray:
    #print("(1024,2)", z0.shape)
    #print("should be (201,) array", ab.shape, )

    # add the gaussian noise

    noise = np.random.randn(*z0.shape)

    alpha_sqrt = np.sqrt(ab[k])
    alpha_sqrt_second = np.sqrt(1- ab[k])

    zk = alpha_sqrt * z0 + alpha_sqrt_second * noise

    return zk.astype(np.float32) # zk = sqrt(ab) * z0 + sqrt(1-ab)*noise which is SCHEDULED NOISE in a sense
    #raise NotImplementedError


def train_score(Zs, cfg: LatentConfig, model: nn.Module, device: str) -> None:
    # Train score model with Hyvärinen objective
    ab = alpha_bar(cfg.K, cfg.beta0, cfg.beta1)

    # TODO: implement the optimizer
    opt = torch.optim.Adam(model.parameters(), cfg.lr)
    model.train()
    loss_list = []
    div_list = []

    bar = tqdm(range(1, cfg.steps + 1))
    for step in range(1, cfg.steps + 1):
        idx = np.random.randint(0, Zs.shape[0], size=cfg.batch_size)
        z0 = Zs[idx]

        k = np.random.randint(1, cfg.K + 1)
        tval = k / cfg.K

        zk = add_noise(z0, k, ab, cfg)  # noised latents at time k

        z = torch.tensor(zk, device=device, requires_grad=True)
        t = torch.full((cfg.batch_size, 1), float(tval), device=device)

        s = model(z, t)
        div = divergence_hutchinson(s, z)

        # TODO: implement Hyvärinen loss and optimization step
        loss = 0.5 * (s.norm(dim=1) ** 2).mean() + div.mean()
   
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_list.append(loss.item())
        div_list.append(div.mean())
        bar.set_description(f"[train] step {step}/{cfg.steps} loss={loss.item():.6f}")
        bar.update(1)

    bar.close()

    plt.figure(figsize=(6, 4))
    plt.plot(loss_list, lw=0.65, color="grey")
    plt.title("Hyvärinen Loss over training")
    plt.xlabel("Training steps")
    plt.ylabel("Hyvärinen Loss")
    plt.savefig(f"{cfg.outdir}/training_loss.png", dpi=200)
    plt.close()

    print("hello")

    plt.figure(figsize=(6, 4))
    plt.plot(div_list, lw=0.65, color="grey")
    plt.title("Mean divergence over training")
    plt.xlabel("Training steps")
    plt.ylabel("Divergence")
    plt.savefig(f"{cfg.outdir}/training_divergence.png", dpi=200)
    plt.close()


def save_grid(imgs: torch.Tensor, path: str, title: str | None = None, nrow: int = 8):
    imgs = imgs.detach().cpu().clamp(0, 1)
    n = imgs.shape[0]
    ncol = math.ceil(n / nrow)
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow, ncol))
    axes = np.array(axes).reshape(ncol, nrow)
    idx = 0
    for r in range(ncol):
        for c in range(nrow):
            axes[r, c].axis("off")
            if idx < n:
                axes[r, c].imshow(imgs[idx, 0], cmap="gray")
            idx += 1
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    cfg = LatentConfig()
    os.makedirs(cfg.outdir, exist_ok=True)
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # Load VAE
    vae = VAE(
        input_dim=28 * 28, hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim
    ).to(device)
    vae.load_state_dict(torch.load(cfg.vae_weights, map_location=device))
    vae.eval()

    # Load MNIST
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(root=cfg.outdir, train=True, download=True, transform=tfm)
    loader = DataLoader(train_ds, batch_size=256, shuffle=False)

    # P3.3.2 build latent dataset (z = mu(x))
    Z, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            x_flat = x.view(x.size(0), -1)
            mu, logvar = vae.enc(x_flat)
            if mu is not None:
                Z.append(mu.cpu().numpy())
                Y.append(y.numpy())

    if len(Z) > 0:
        Z = np.concatenate(Z, axis=0).astype(np.float32)
        Y = np.concatenate(Y, axis=0).astype(np.int64)
        print("latent dataset:", Z.shape)

        # standardize (recommended)
        z_mean = Z.mean(axis=0, keepdims=True)
        z_std = Z.std(axis=0, keepdims=True) + 1e-6
        Zs = (Z - z_mean) / z_std

        if cfg.latent_dim == 2:
            plt.figure(figsize=(5, 5))
            plt.scatter(Zs[:, 0], Zs[:, 1], s=2, c=Y, cmap="tab10", alpha=0.7)
            plt.title("Latent dataset (standardized), dz=2")
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(os.path.join(cfg.outdir, "latent_scatter_dz2.png"), dpi=200)
            plt.close()

    
    # P3.3.3 train score model in latent space
    model = ScoreNet(hidden=cfg.score_hidden).to(device)

    # TODO: implement train_score and uncomment:
    train_score(Zs, cfg, model, device)

    
    
    # P3.3.4 sample latents with reverse process + decode
    def score_learned(z_np: np.ndarray, k: int) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            zt = torch.tensor(z_np, device=device)
            tt = torch.full((z_np.shape[0], 1), float(k / cfg.K), device=device)
            return model(zt, tt).cpu().numpy().astype(np.float32)

    # Sample in latent space
    zK = np.random.randn(64, cfg.latent_dim).astype(np.float32)  # start from N(0,I)
    zs_traj = reverse_euler(cfg, zK, score_learned)
    if zs_traj is not None:
        z0_gen = zs_traj[0]  # final

        # un-standardize back to VAE latent coordinates
        z0_gen = z0_gen * z_std + z_mean

    # TODO: decode z0_gen to images with VAE decoder

    x_hat = vae.dec(torch.from_numpy(z0_gen).to(device))
    x_hat = x_hat.view(-1, 1, 28, 28)
    save_grid(x_hat, "outputs_P3.3_latent/latent_diff_reconstruction.png", title="Generation of samples (drawn from latent space)")

    # P3.3.5 compare to VAE-only
    # TODO: generate samples from VAE-only (z ~ N(0,I) decode)

    x_hat = vae.dec(torch.from_numpy(zK).to(device))
    x_hat = x_hat.view(-1, 1, 28, 28)
    save_grid(x_hat, "outputs_P3.3_latent/normal_drawn_reconstruction.png", title="Generation of samples (drawn from multivariate normal distribution with N(0, I))")

    # TODO: you can create a combined comparison figure in your report

    # Save model weights
    torch.save(model.state_dict(), os.path.join(cfg.outdir, "latent_score_net.pt"))
    print("Saved weights:", os.path.join(cfg.outdir, "latent_score_net.pt"))
    


if __name__ == "__main__":
    main()
