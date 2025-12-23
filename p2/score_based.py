#!/usr/bin/env python3
"""
P3 (Mini project) — Code template 2/3: Score-based modeling on a 2D spiral (P3.2)

This skeleton covers:
- P3.2.1 forward VP-SDE (Euler–Maruyama)
- P3.2.2 reverse process using the TRUE score (mixture score)  [TODO]
- P3.2.3 learn score with Hyvärinen objective + Hutchinson divergence  [TODO divergence helper]
- P3.2.4 generation with learned score

Students must fill:
- true mixture score (responsibilities)
- Hutchinson divergence helper
- score net design tweaks and reporting/plots

Run:
  python P3_2_spiral_score_template.py
"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class DiffusionConfig:
    # Spiral mixture
    M: int = 200
    a: float = 0.2
    b: float = 0.15
    theta_min: float = 0.0
    theta_max: float = 4.0 * math.pi
    sigma0: float = 0.06

    # VP schedule (exam)
    beta0: float = 0.001
    beta1: float = 0.2
    K: int = 301  # discretization steps for SDE.    # TODO: try different K

    # Score learning
    steps: int = 5000  # noising steps
    batch_size: int = 64  # TODO: choose
    lr: float = 1e-3  # TODO: choose

    outdir: str = "outputs_q2_spiral"
    seed: int = 0


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def spiral_means(cfg: DiffusionConfig) -> np.ndarray:
    thetas = np.linspace(cfg.theta_min, cfg.theta_max, cfg.M)
    r = cfg.a + cfg.b * thetas
    mu = np.stack([r * np.cos(thetas), r * np.sin(thetas)], axis=1).astype(np.float32)
    return mu


def sample_p0(cfg: DiffusionConfig, n: int) -> np.ndarray:
    mu = spiral_means(cfg)
    idx = np.random.randint(0, cfg.M, size=n)
    eps = np.random.randn(n, 2).astype(np.float32) * cfg.sigma0
    return mu[idx] + eps


def beta(t: float, beta0, beta1) -> float:
    # Linear schedule
    return beta0 + (beta1 - beta0) * t


def alpha_bar(K, beta0, beta1) -> np.ndarray:
    # Precompute alpha_bar for k=0,...,K
    
    t = 0
    delta_t = 1 / K
    alpha_bar = np.zeros(K + 1)
    alpha_bar[0] = np.exp(-beta(t, beta0, beta1) * delta_t)

    for i in range(1, K+1):
        t += delta_t
        alpha_bar[i] = alpha_bar[i-1] * np.exp(-beta(t, beta0, beta1) * delta_t)

    return alpha_bar


def forward_euler(
    cfg: DiffusionConfig, x0: np.ndarray
) -> Tuple[List[np.ndarray], np.ndarray]:
    # Forward Euler–Maruyama to simulate VP-SDE
    # TODO: implement the forward process

    t = 0
    dim = x0.shape[1] #should just be a dimension of 2
    delta_t = 1 / cfg.K
    xk = x0.copy()
    
    at_step_k = [x0]

    for i in range(1, cfg.K):
 
        noise = np.random.multivariate_normal(mean=[0, 0], cov=np.eye(dim), size=x0.shape[0])

        beta_val = beta(t, cfg.beta0, cfg.beta1)

        x_next = xk - 0.5 * beta_val * xk * delta_t + np.sqrt(beta_val * delta_t) * noise
        at_step_k.append(x_next)

        xk = x_next.copy()
        t += delta_t

    return (at_step_k, xk)


def pk_mixture_params(cfg: DiffusionConfig, k: int) -> Tuple[np.ndarray, float]:
    # Compute p_k mixture parameters at step k
    # TODO: implement the pk parameters

    alpha_k = alpha_bar(cfg.K, cfg.beta0, cfg.beta1)[k-1]
    mu = spiral_means(cfg)

    means_k = np.sqrt(alpha_k) * mu

    var_k = alpha_k * cfg.sigma0**2 + 1 - alpha_k

    return means_k.astype(np.float32), var_k


def true_score_mixture(x: np.ndarray, means: np.ndarray, var: float) -> np.ndarray:
    """
    TODO: implement score of spiral distribution at step k:
      p(x) = (1/M) sum_m N(x|mu_m, var I)
      score = ∇ log p(x) = sum_m w_m(x) * (-(x-mu_m)/var)
    """
    # TODO 

    N, dim = x.shape
    M = means.shape[0]

    diff = x[:, np.newaxis] - means[np.newaxis, :, :]
    exponent = -0.5 * np.sum(diff**2, axis=2) /var

    exp = np.exp(exponent)

    grad_components = exp[:, :, np.newaxis] * (-diff / var)
    numerator = np.sum(grad_components, axis=1)
    denominator = np.sum(exp, axis=1)[:, np.newaxis]

    score = numerator / denominator

    return score


# P3.2.2 reverse with TRUE score
def score_true(x: np.ndarray, k: int) -> np.ndarray:
    # TODO: implement the true_score

    cfg = DiffusionConfig()

    means, var = pk_mixture_params(cfg, k)

    return true_score_mixture(x, means, var)

def score_learned(x: np.ndarray, k: int) -> np.ndarray:
    cfg = DiffusionConfig()

    

    return 

def reverse_euler(cfg: DiffusionConfig, xK: np.ndarray, score_fn) -> List[np.ndarray]:
    # Reverse Euler–Maruyama using given score function
    # TODO: implement the reverse process


    delta_t = 1 / cfg.K
    xK = xK.copy()
    dim = xK.shape[1]

    at_step_k = [xK.copy()]

    for k in range(cfg.K, 0, -1):
        t = k / cfg.K 
        beta_val = beta(t, cfg.beta0, cfg.beta1)
        de_noise = np.random.randn(*xK.shape) # Standard normal
        
        # Corrected Drift sign: counter-act the inward pull of the forward SDE
        xK += (-0.5 * beta_val * xK + beta_val * score_fn(xK, k)) * delta_t + np.sqrt(beta_val * delta_t) * de_noise

        at_step_k.append(xK.copy())
    '''
    for k in range(cfg.K, 0, -1):
        t = k / cfg.K 
        beta_val = beta(t, cfg.beta0, cfg.beta1)

        de_noise = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=xK.shape[0])

        xK += (-0.5 * beta_val * xK + beta_val * score_fn(xK, k)) * delta_t + np.sqrt(beta_val * delta_t) * de_noise

        at_step_k.append(xK.copy())
        '''
    
    return at_step_k[::-1]


class ScoreNet(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        # TODO: design your score network architecture
        self.sequential = nn.Sequential(
            nn.Linear(3, hidden), #two input dims and time
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2), # just two putput dims
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # TODO: implement forward pass to output score
        score = self.sequential(torch.cat([x, t],dim=1))
        return score


def divergence_hutchinson(s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Exact divergence for 2D vector field v(x) using autograd:
      div v = d v1/dx1 + d v2/dx2
    v: (B,2), x: (B,2) with requires_grad=True
    returns: (B,)
    """
    v1 = s[:, 0]
    v2 = s[:, 1]
    grads1 = torch.autograd.grad(v1.sum(), x, create_graph=True)[0][:, 0]
    grads2 = torch.autograd.grad(v2.sum(), x, create_graph=True)[0][:, 1]
    return grads1 + grads2


def train_score(cfg: DiffusionConfig, model: nn.Module, device: str) -> None:
    # Train score model with Hyvärinen objective

    # TODO: implement the optimizer
    opt = torch.optim.Adam(model.parameters(), cfg.lr) 
    model.train()
    loss_list = []

    bar = tqdm(range(1, cfg.steps + 1))
    for step in range(1, cfg.steps + 1):
        k = np.random.randint(1, cfg.K + 1)

        tval = k / cfg.K
        t = torch.full((cfg.batch_size, 1), float(tval), device=device)

        means_k, var_k = pk_mixture_params(cfg, k)
        m_idx = np.random.randint(0, cfg.M, size=cfg.batch_size)
        xk = means_k[m_idx] + np.random.randn(cfg.batch_size, 2).astype(
            np.float32
        ) * math.sqrt(var_k)

        x = torch.tensor(xk, device=device, requires_grad=True)
        

        s = model(x, t)  # score estimate
        div = divergence_hutchinson(s, x)  # divergence estimate

        # TODO: implement Hyvärinen loss and optimization step
        loss = 0.5 * (s.norm(dim=1) ** 2).mean() + div.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_list.append(loss.item())
        bar.set_description(f"[train] step {step}/{cfg.steps} loss={loss.item():.6f}")
        bar.update(1)

    bar.close()

    plt.figure(figsize=(6, 4))
    plt.plot(loss_list, lw=0.65, color="grey")
    plt.title("Loss over training")
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.savefig(f"{cfg.outdir}/training.png", dpi=200)
    plt.close()



def scatter(x: np.ndarray, title: str, path: str):
    plt.figure(figsize=(4, 4))
    plt.scatter(x[:, 0], x[:, 1], s=4, alpha=0.6)
    plt.title(title)
    plt.axis("equal")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


'''
just a function to grid-plot 6 figures
'''
def plot_at_k(at_step_k, k_steps, extra=""): 
    if len(k_steps) != 6:
        raise ValueError("Must be 6 time steps")

    os.makedirs("k_step", exist_ok=True) 
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 12)) 
    axes = axes.flatten() 
    
    for idx, k in enumerate(k_steps): 
        xk = at_step_k[k] 

        ax = axes[idx] 
        ax.scatter(xk[:, 0], xk[:, 1], s=4, alpha=0.6) 
        ax.set_title(f"k = {k}") 
        ax.set_aspect("equal") 
        ax.grid(alpha=0.2) 
    plt.tight_layout() 
    plt.savefig("k_step/" + extra + "grid_plot.png", dpi=200) 
    plt.close()

'''
plotting the learned vector field
'''
def grid_2d_vector(model, device, t_vals, lim=2.5):
    if len(t_vals) != 6:
        raise ValueError("Must be 6 time steps")
    os.makedirs("vector_grid", exist_ok=True) 

    model.eval()

    fig, axes = plt.subplots(3, 2, figsize=(10, 12)) 
    axes = axes.flatten() 
    
    for idx, t_val in enumerate(t_vals): 
        xs = np.linspace(-lim, lim, 30)
        ys = np.linspace(-lim, lim, 30)
        
        X,Y = np.meshgrid(xs, ys)

        grid = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
        x = torch.tensor(grid, device=device)

        t = torch.full((x.shape[0], 1), float(t_val), device=device)

        with torch.no_grad():
            s= model(x, t).cpu().numpy()

        U = s[:, 0].reshape(30,30)
        V = s[:, 1].reshape(30,30)

        ax = axes[idx]
        ax.set_title(f"Learned Vector Field for t={t_val:3f}") 
        ax.set_aspect("equal") 
        ax.grid(alpha=0.2) 
        ax.quiver(X, Y, U, V, color="grey")


    plt.tight_layout() 
    plt.savefig("vector_grid/grid_plot.png", dpi=200) 
    plt.close()

'''
True vs denoised
'''
def comparison_plot(x_true: np.ndarray, x_denoise: np.ndarray = None, x_approx :np.ndarray = None, save="k_step/"):
    approx = ""
    plt.figure(figsize=(6, 6))
    plt.scatter(x_true[:, 0], x_true[:, 1], s=4, alpha=0.5, color="blue", label="True Distribution")
    if x_denoise is not None:
        plt.scatter(x_denoise[:, 0], x_denoise[:, 1], s=4, alpha=0.5, color="orange", label="Denoised Distribution")
    if x_approx is not None:
        plt.scatter(x_approx[:, 0], x_approx[:, 1], s=4, alpha=0.5, color="purple", label="Learned Denoise Distribution")
        approx = "approx_"
    plt.title("True vs Denoised Distributions")
    plt.axis("equal")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save + approx + "comparison_plot.png", dpi=200)
    plt.close()


def main():
    cfg = DiffusionConfig()
    os.makedirs(cfg.outdir, exist_ok=True)
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # P3.2.1 forward
    x0 = sample_p0(cfg, n=4000)

    plt.figure(figsize=(6, 6))
    plt.scatter(x0[:2500, 0], x0[:2500, 1], s=4, alpha=0.6)
    plt.title("Data x0")
    plt.axis("equal")
    plt.grid(alpha=0.2)
    # plt.tight_layout()
    plt.savefig(os.path.join(cfg.outdir, "Spiral distribution.png"), dpi=200)
    plt.close()

    (at_step_k, final) = forward_euler(cfg, x0)
    plot_at_k(at_step_k, [0, 60, 120, 180, 240, 300])

    # TODO: implement true_score_mixture, run:
    samp, dim = x0.shape
    x_normal = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=samp)
    xs_rev_true = reverse_euler(cfg, x_normal.copy(), score_true)
    plot_at_k(xs_rev_true, [300, 240, 180, 120, 60, 0])
    #for k in [300, 240, 180, 30, 0]:
       # scatter(xs_rev_true[k], f"reverse(true) k={k}", os.path.join(cfg.outdir, f"rev_true_{k}.png"))

    comparison_plot(x0, xs_rev_true[0])

    
    # P3.2.3 learn score (students paste divergence helper first)
    model = ScoreNet(hidden=256).to(device)
    # TODO: implement train score and uncomment:
    train_score(cfg, model, device)
    grid_2d_vector(model, device, [1 / cfg.K, 0.1, 0.35, 0.65, 0.90, 1-1/cfg.K])

    # P3.2.4 reverse with learned score (after training)
    # TODO: define score_learned 
    def score_learned(x, k):
        model.eval()
        x_tensor = torch.from_numpy(x).to(torch.float32).to(device)
        t_val = k / cfg.K
        t_tensor = torch.ones((x.shape[0], 1)) * t_val
        t_tensor = torch.full((x.shape[0], 1), t_val, device=device, dtype=torch.float32)

        with torch.no_grad():
            score = model(x_tensor, t_tensor)
        return score.cpu().numpy()

    # TODO: run reverse_euler(cfg, xK, score_learned) and compare plots
    xs_rev_approx = reverse_euler(cfg, x_normal.copy(), score_learned)
    
    plot_at_k(xs_rev_approx, [300, 240, 180, 120, 60, 0], extra="learned_")
    comparison_plot(x0, xs_rev_true[0], xs_rev_approx[0])

    # Save model weights
    torch.save(model.state_dict(), os.path.join(cfg.outdir, "score_net_spiral.pt"))
    print("Saved weights:", os.path.join(cfg.outdir, "score_net_spiral.pt"))
    


if __name__ == "__main__":
    main()
