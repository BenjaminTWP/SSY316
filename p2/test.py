
from score_based import alpha_bar, DiffusionConfig, ScoreNet
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


#print(alpha_bar(400, 0.001, 0.2))


cfg = DiffusionConfig()

model = ScoreNet(hidden=64)

model.load_state_dict(torch.load("outputs_q2_spiral/score_net_spiral.pt", weights_only=True, map_location=torch.device('cpu')))
print(model.eval())

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


def plot_at_k(at_step_k, k_steps): 
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
    plt.savefig("k_step/grid_plot.png", dpi=200) 
    plt.close()


grid_2d_vector(model, "cpu", [1 / cfg.K, 0.1, 0.35, 0.65, 0.90, 1-1/cfg.K])