from train_vae import VAE, VAEConfig
from torchinfo import summary

cfg = VAEConfig()
vae = VAE(28 * 28, cfg.hidden_dim, cfg.latent_dim)
summary(vae, (1, 1, 784))