#!/usr/bin/env python3
"""
P3 (Mini project) — Code template 1/3: VAE for MNIST (P3.1)

This file is a minimal PyTorch skeleton that students can COMPLETE.
It includes only what students need (data loading + structure + TODOs).

Students must fill:
- Encoder/decoder architecture choices
- ELBO (negative ELBO) details
- Training curves + plots
- Generation grid

Run:
  python P3_1_vae_template.py
"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class VAEConfig:
    latent_dim: int = 2  # TODO: choose dz
    hidden_dim: int = 256  # TODO: choose
    batch_size: int = 64  # TODO: choose
    epochs: int = 150  # TODO: choose
    lr: float = 2e-3  # TODO: choose
    use_bce: bool = True  # Bernoulli decoder (recommended for MNIST)
    outdir: str = "outputs_q1_vae"
    seed: int = 0


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Encoder(nn.Module):
    """q_phi(z|x) = N(mu(x), diag(sigma(x)^2))"""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement forward pass to output mu and logvar
        x = self.encoder(x)
        mu = self.mu(x)  # replace with actual computation
        logvar = self.log_var(x)  # replace with actual computation
        return mu, logvar


class Decoder(nn.Module):
    """p_theta(x|z)"""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # TODO: Define Decoder architecture
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, output_dim),
                                     nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass to output logits
        logits = self.decoder(z)  # replace with actual computation
        return logits


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.enc = Encoder(input_dim, hidden_dim, latent_dim)
        self.dec = Decoder(latent_dim, hidden_dim, input_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.enc(x)
        z = self.reparameterize(mu, logvar)
        logits = self.dec(z)
        return logits, mu, logvar


def negative_elbo(
    x: torch.Tensor,
    logits: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    use_bce: bool,
):
    """
    Negative ELBO = reconstruction + KL.
    """
    # ToDO: Implement negative ELBO computation

    if use_bce:
        bce_loss = nn.BCELoss(reduction='sum')
        recon = bce_loss(logits, x) # torch.tensor(0, requires_grad=True)  # replace with actual computation


    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar)) # torch.tensor(0, requires_grad=True)  # replace with actual computation

    total = recon + kl
    return total, recon, kl


def save_image_grid(
    x: torch.Tensor, path: str, nrow: int = 8, title: str | None = None
):
    x = x.detach().cpu().clamp(0, 1)
    n = x.shape[0]
    ncol = math.ceil(n / nrow)
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow, ncol))
    axes = axes.reshape(ncol, nrow)
    idx = 0
    for r in range(ncol):
        for c in range(nrow):
            axes[r, c].axis("off")
            if idx < n:
                print("shape of x:", x.shape)
                axes[r, c].imshow(x[idx, 0], cmap="gray")
            idx += 1
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=400)
    plt.close(fig)


def main():
    cfg = VAEConfig()
    os.makedirs(cfg.outdir, exist_ok=True)
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(root=cfg.outdir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=cfg.outdir, train=False, download=True, transform=tfm)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    vae = VAE(28 * 28, cfg.hidden_dim, cfg.latent_dim).to(device)
    # TODO: implement optimizer
    optimizer = torch.optim.Adam(vae.parameters(), cfg.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # TODO: track curves (loss / recon / KL) for plots
    bar = tqdm(range(1, cfg.epochs + 1))
    loss_list, recon_list, kl_list = [], [], []
    for epoch in range(1, cfg.epochs + 1):
        vae.train()
        total_loss, total_recon, total_kl = 0, 0, 0
        for x, _ in train_loader:
            x = x.to(device)
            x_flat = x.view(x.size(0), -1)
            # TODO: implement training loop

            optimizer.zero_grad()

            logits, mu, log_var = vae(x_flat)

            loss, recon, kl = negative_elbo(x_flat, logits, mu, log_var, cfg.use_bce)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
        
        scheduler.step()

        epoch_loss = total_loss / len(train_loader.dataset)
        loss_list.append(epoch_loss)
        avg_recon = total_recon / len(train_loader.dataset)
        recon_list.append(avg_recon)
        avg_kl = total_kl / len(train_loader.dataset)
        kl_list.append(avg_kl)

        bar.set_description(
            f"epoch {epoch}/{cfg.epochs} done. Epoch loss: {epoch_loss}, reconstruction: {avg_recon}, KL: {avg_kl}"
        )
        bar.update(1)

    plt.plot(loss_list)
    plt.title("Negative ELBO (loss) over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Negative ELBO (loss) per epoch")
    plt.grid()
    plt.savefig("outputs_q1_vae/neg_elbo_plot")
    plt.show()

    plt.figure() 
    plt.plot(recon_list, kl_list)
    plt.title("Reconstruction VS. KL")
    plt.xlabel("Reconstruction")
    plt.ylabel("KL")
    plt.grid()
    plt.savefig("outputs_q1_vae/recon_vs_Kl")
    plt.show()

    # P3.1.4 generation
    # TODO: generate samples from prior and save grid
    mean = torch.zeros(cfg.latent_dim, device=device)
    cov = torch.eye(cfg.latent_dim, device=device)
    dist = torch.distributions.MultivariateNormal(mean, cov)
    z = dist.sample((32,)).to(device)

    x_hat = vae.dec(z)
    x_hat = x_hat.view(-1, 1, 28, 28)
    save_image_grid(x_hat, "outputs_q1_vae/example_reconstruction.png", title="Generation of samples (drawn from multivariate normal distribution with N(0, I))")

    # Save model weights
    torch.save(vae.state_dict(), os.path.join(cfg.outdir, "vae_mnist.pt"))
    print("Saved weights:", os.path.join(cfg.outdir, "vae_mnist.pt"))


if __name__ == "__main__":
    main()