from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm

from NIR_II_Guided_Diffusion.data.datasets.utils import MatDataset
from scipy.io import loadmat

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.functional import multiscale_structural_similarity_index_measure as mmssim
from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio

from NIR_II_Guided_Diffusion.models.embedders.latent_embedders import VAE, VAEGAN, VQVAE, VQGAN
from config import get_config


# ----------------Settings --------------
batch_size = 100
max_samples = None  # set to None for all
target_class = None  # None for no specific class

path_out = Path.cwd() / 'results' / 'metrics'
path_out.mkdir(parents=True, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------- Logging -----------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(path_out / f'metrics_{current_time}.log', 'w'))


# -------------- Helpers ---------------------
pil2torch = lambda x: torch.as_tensor(np.array(x)).moveaxis(-1, 0)  # In contrast to ToTensor(), this will not cast 0-255 to 0-1 and destroy uint8 (required later)

# ---------------- Load Dataset ----------------

EvaluateFMT = loadmat('__your__dataset__path')  # Replace with your dataset path
test_dataset = MatDataset(EvaluateFMT)
ds_real = test_dataset

dm_real = DataLoader(ds_real, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)

logger.info(f"Samples Real: {len(ds_real)}")


# --------------- Load Model ------------------
config = get_config()
encoder_type = config.encoder_type

if encoder_type == 'VAE':
    model = VAE.load_from_checkpoint('__your__Encoder__checkpoint__path')  # Replace with your encoder checkpoint path
elif encoder_type == 'VAEGAN':
    model = VAEGAN.load_from_checkpoint('__your__Encoder__checkpoint__path')  # Replace with your encoder checkpoint path
elif encoder_type == 'VQVAE':
    model = VQVAE.load_from_checkpoint('__your__Encoder__checkpoint__path')  # Replace with your encoder checkpoint path
elif encoder_type == 'VQGAN':
    model = VQGAN.load_from_checkpoint('__your__Encoder__checkpoint__path')  # Replace with your encoder checkpoint path
else:
    raise ValueError(f"Invalid encoder_type: {encoder_type}")

model.to(device)

# ------------- Initialize Metrics ----------------------
calc_lpips = LPIPS().to(device) if 'LPIPS' in config.metrics else None

# --------------- Start Calculation -----------------
mmssim_list, mse_list, ssim_list, psnr_list = [], [], [], []
for real_batch in tqdm(dm_real):
    imgs_real_batch = real_batch.to(device)
    imgs_real_batch = imgs_real_batch.repeat(1, 3, 1, 1)

    with torch.no_grad():
        imgs_fake_batch = model(real_batch)[0].clamp(-1, 1)
        imgs_fake_batch = imgs_fake_batch.repeat(1, 3, 1, 1)

    # -------------- Metrics Calculation -------------------
    for img_real, img_fake in zip(imgs_real_batch, imgs_fake_batch):
        img_real, img_fake = (img_real + 1) / 2, (img_fake + 1) / 2  # [-1, 1] -> [0, 1]

        if 'MS_SSIM' in config.metrics:
            mmssim_val = mmssim(img_real[None], img_fake[None], normalize='relu', kernel_size=5, betas=(0.4, 0.3, 0.3))
            mmssim_list.append(mmssim_val)

        if 'MSE' in config.metrics:
            mse_val = torch.mean(torch.square(img_real - img_fake))
            mse_list.append(mse_val)

        if 'SSIM' in config.metrics:
            ssim_val = structural_similarity_index_measure(img_real[None], img_fake[None])
            ssim_list.append(ssim_val)

        if 'PSNR' in config.metrics:
            psnr_val = peak_signal_noise_ratio(img_real[None], img_fake[None])
            psnr_list.append(psnr_val)

    if 'LPIPS' in config.metrics:
        calc_lpips.update(imgs_real_batch, imgs_fake_batch)

# -------------- Summary -------------------
    if 'MS_SSIM' in config.metrics:
        mmssim_list = torch.stack(mmssim_list)
    if 'MSE' in config.metrics:
        mse_list = torch.stack(mse_list)
    if 'SSIM' in config.metrics:
        ssim_list = torch.stack(ssim_list)
    if 'PSNR' in config.metrics:
        psnr_list = torch.stack(psnr_list)

    if 'LPIPS' in config.metrics:
        lpips = 1 - calc_lpips.compute()
        logger.info(f"LPIPS Score: {lpips}")
    if 'MS_SSIM' in config.metrics:
        logger.info(f"MS-SSIM: {torch.mean(mmssim_list)} ± {torch.std(mmssim_list)}")
    if 'MSE' in config.metrics:
        logger.info(f"MSE: {torch.mean(mse_list)} ± {torch.std(mse_list)}")
    if 'SSIM' in config.metrics:
        logger.info(f"SSIM: {torch.mean(ssim_list)} ± {torch.std(ssim_list)}")
    if 'PSNR' in config.metrics:
        logger.info(f"PSNR: {torch.mean(psnr_list)} ± {torch.std(psnr_list)}")