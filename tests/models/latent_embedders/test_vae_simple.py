import torch 
from NIR_II_Guided_Diffusion.models.embedders.latent_embedders import VAE


input = torch.randn((1, 3, 128, 128)) # [B, C, H, W]


model = VAE(in_channels=3, out_channels=3, spatial_dims = 2, deep_supervision=True)
output = model(input)
print(output)


