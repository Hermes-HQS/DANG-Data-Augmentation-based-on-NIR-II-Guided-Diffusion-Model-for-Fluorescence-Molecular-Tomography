from email.mime import audio
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import torchio as tio
from scipy.io import loadmat

from NIR_II_Guided_Diffusion.data.datamodules import SimpleDataModule
from NIR_II_Guided_Diffusion.models.pipelines import DiffusionPipeline
from NIR_II_Guided_Diffusion.models.estimators import UNet
from NIR_II_Guided_Diffusion.external.stable_diffusion.unet_openai import UNetModel
from NIR_II_Guided_Diffusion.models.noise_schedulers import GaussianNoiseScheduler
from NIR_II_Guided_Diffusion.models.embedders import LabelEmbedder, TimeEmbbeding
from NIR_II_Guided_Diffusion.models.embedders.latent_embedders import VAE, VAEGAN, VQVAE, VQGAN

from NIR_II_Guided_Diffusion.data.datasets.utils import MatDataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from config import get_config


if __name__ == "__main__":
    # ------------ Load Data ----------------
    TrainFMT = loadmat('TrainingImage2.mat')
    train_dataset = MatDataset(TrainFMT)

    dm = SimpleDataModule(
        ds_train=train_dataset,
        batch_size=8,
        num_workers=0,
        pin_memory=True
    )

    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    # ------------ Config --------------------
    config = get_config()

    # ------------ Initialize Model ------------
    cond_embedder = LabelEmbedder
    cond_embedder_kwargs = {
        'emb_dim': config.cond_emb_dim,
        'num_classes': config.num_classes
    }

    time_embedder = TimeEmbbeding
    time_embedder_kwargs = {
        'emb_dim': config.time_emb_dim
    }

    noise_estimator = UNet
    noise_estimator_kwargs = {
        'in_ch': config.unet_in_ch,
        'out_ch': config.unet_out_ch,
        'spatial_dims': config.spatial_dims,
        'hid_chs': config.unet_hid_chs,
        'kernel_sizes': config.unet_kernel_sizes,
        'strides': config.unet_strides,
        'time_embedder': time_embedder,
        'time_embedder_kwargs': time_embedder_kwargs,
        'cond_embedder': cond_embedder,
        'cond_embedder_kwargs': cond_embedder_kwargs,
        'deep_supervision': config.unet_deep_supervision,
        'use_res_block': config.unet_use_res_block,
        'use_attention': config.unet_use_attention,
    }

    # ------------ Initialize Noise ------------
    noise_scheduler = GaussianNoiseScheduler
    noise_scheduler_kwargs = {
        'timesteps': config.noise_timesteps,
        'beta_start': config.noise_beta_start,
        'beta_end': config.noise_beta_end,
        'schedule_strategy': config.noise_schedule_strategy
    }

    # ------------ Initialize Latent Space  ------------
    latent_embedder = VAE
    latent_embedder_checkpoint = 'runs/20250514/last.ckpt'

    # ------------ Initialize Pipeline ------------
    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator,
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler,
        noise_scheduler_kwargs=noise_scheduler_kwargs,
        latent_embedder=latent_embedder,
        latent_embedder_checkpoint=latent_embedder_checkpoint,
        estimator_objective='x_T',
        estimate_variance=False,
        use_self_conditioning=False,
        use_ema=False,
        classifier_free_guidance_dropout=0.5,  # Disable during training by setting to 0
        do_input_centering=False,
        clip_x0=False,
        sample_every_n_steps=1000
    )

    # -------------- Training Initialization ---------------
    to_monitor = "train/loss"  # "pl/val_loss"
    min_max = "min"
    save_and_sample_every = 100

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0,  # minimum change in the monitored quantity to qualify as an improvement
        patience=30,  # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),  # dirpath
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=2,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator=accelerator,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every,
        limit_val_batches=0,  # 0 = disable validation - Note: Early Stopping no longer available
        min_epochs=100,
        max_epochs=1001,
        num_sanity_val_steps=2,
    )

    # ---------------- Execute Training ----------------
    trainer.fit(pipeline, datamodule=dm)

    # ------------- Save path to best model -------------
    pipeline.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)


