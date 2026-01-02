#!/usr/bin/env python3

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

from P3_1_vae_template import ( VAE )

from P3_2_spiral_score_template import (
    divergence_hutchinson,
    reverse_euler,
    alpha_bar,
)

from P3_3_latent_diffusion_template import ( add_noise, save_grid )


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class GenerateConfig:
    vae_weights: str = (
        "outputs_q1_vae/vae_mnist.pt"  # where VAE is
    )
    latent_dim: int = 2  # will match vae implementation
    hidden_dim: int = 256  # will match vae implementation

    score_hidden: int = 256  # chosen

    beta0: float = 0
    beta1: float = 0.2
    K: int = 100 #####

    steps: int = 10000
    batch_size: int = 256
    lr: float = 2e-4

    outdir: str = "outputs_q4_generate"
    seed: int = 0


class ScoreNet(nn.Module):
    def __init__(self, cfg: GenerateConfig, num_digits=10):
        super().__init__()

        self.label_embedding = nn.Embedding(num_digits, cfg.score_hidden)

        self.time_emb = nn.Sequential(
            nn.Linear(1, cfg.score_hidden),
            nn.SiLU()
        )

        self.first_layer = nn.Linear(cfg.latent_dim, cfg.score_hidden) #latent dim just 2


        self.sequential = nn.Sequential(
            nn.Linear(cfg.score_hidden * 3, cfg.score_hidden),
            nn.SiLU(),
            nn.Linear(cfg.score_hidden, cfg.score_hidden),
            nn.SiLU(),
            nn.Linear(cfg.score_hidden, cfg.score_hidden),
            nn.SiLU(),
            nn.Linear(cfg.score_hidden, cfg.latent_dim), 
        )

    def forward(self, z: torch.Tensor, k: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z_first = self.first_layer(z)
        k_time = self.time_emb(k.float())
        y_embedding = self.label_embedding(y)

        all_combined = torch.cat([z_first, k_time, y_embedding], dim=1)
        return self.sequential(all_combined)


def train_score(Z, Y, cfg: GenerateConfig, model: nn.Module, device: str) -> None:
    # Train score model with Hyvärinen objective
    ab = alpha_bar(cfg.K, cfg.beta0, cfg.beta1)

    opt = torch.optim.Adam(model.parameters(), cfg.lr)
    model.train()
    loss_list = []
    div_list = []

    bar = tqdm(range(1, cfg.steps + 1))
    for step in range(1, cfg.steps + 1):
        idx = np.random.randint(0, Z.shape[0], size=cfg.batch_size)
        z0 = torch.tensor(Z[idx], device=device, dtype=torch.float32)
        y = torch.tensor(Y[idx], device=device, dtype=torch.long)

        k_vals = np.random.randint(1, cfg.K + 1, size=(cfg.batch_size, 1))
        k = torch.tensor(k_vals, device=device, dtype=torch.float32)

        zk = add_noise(z0.cpu().numpy(), k_vals, ab, cfg) 
        z = torch.tensor(zk, device=device, dtype=torch.float32, requires_grad=True)

        s = model(z, k, y)
        div = divergence_hutchinson(s, z)

        loss = 0.5 * (s.norm(dim=1) ** 2).mean() + div.mean()
   
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_list.append(loss.item())
        div_list.append(div.mean().item())
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

    plt.figure(figsize=(6, 4))
    plt.plot(div_list, lw=0.65, color="grey")
    plt.title("Mean divergence over training")
    plt.xlabel("Training steps")
    plt.ylabel("Divergence")
    plt.savefig(f"{cfg.outdir}/training_divergence.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot([loss_list[i] - div_list[i] for i in range(len(loss_list))], lw=0.65, color="grey")
    plt.title("Loss without divergence over training")
    plt.xlabel("Training steps")
    plt.ylabel("Loss without divergence")
    plt.savefig(f"{cfg.outdir}/training_test.png", dpi=200)
    plt.close()

def main(train: bool = True):
    cfg = GenerateConfig()
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

        # standardize (recommended)
        z_mean = Z.mean(axis=0, keepdims=True)
        z_std = Z.std(axis=0, keepdims=True) + 1e-6
        Zs = (Z - z_mean) / z_std

    print("latent dataset:", Zs.shape)
    print("latent targets:", Y.shape)

    if train:
        model = ScoreNet(cfg).to(device)
        train_score(Zs, Y, cfg, model, device)
    else:
        model = ScoreNet(cfg)
        model.load_state_dict(torch.load(cfg.outdir + "/generate_net.pt", weights_only=True, map_location=torch.device(device)))
        model.to(device)
    
    model.eval()
    DIGIT = 7
    N_SAMPLES = 20

    def score_learned(z_np: np.ndarray, k:int) -> np.ndarray:
        with torch.no_grad():
            zt = torch.tensor(z_np, device=device, dtype=torch.float32)
            kt = torch.full((z_np.shape[0], 1), float(k), device=device)
            yt = torch.full((z_np.shape[0],), DIGIT, device=device, dtype=torch.long)

            return model(zt, kt, yt).cpu().numpy()

    zK = np.random.randn(N_SAMPLES, cfg.latent_dim).astype(np.float32)

    zs_traj = reverse_euler(cfg, zK, score_learned)
    z0_gen = zs_traj[0]
    z0_gen = z0_gen * z_std + z_mean

    with torch.no_grad():
        x_hat = vae.dec(torch.from_numpy(z0_gen).to(device))
        x_hat = x_hat.view(-1, 1, 28, 28)
        save_grid(x_hat, cfg.outdir + "/generated.png", title=f"Generation of digit {DIGIT}", nrow=4)

    if train:
        # Save model weights
        torch.save(model.state_dict(), os.path.join(cfg.outdir, "generate_net.pt"))
        print("Saved weights:", os.path.join(cfg.outdir, "generate_net.pt"))


if __name__ == "__main__":
    main(train=True)