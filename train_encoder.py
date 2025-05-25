from pathlib import Path
from datetime import datetime

from scipy.io import loadmat
import torch 

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateFinder


from NIR_II_Guided_Diffusion.data.datamodules import SimpleDataModule
from NIR_II_Guided_Diffusion.models.embedders.latent_embedders import VQVAE, VQGAN, VAE, VAEGAN
from config import get_config 
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


from NIR_II_Guided_Diffusion.data.datasets.utils import MatDataset


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


if __name__ == "__main__":

    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    gpus = [0] if torch.cuda.is_available() else None


    # ------------ Load Data ----------------
    FMT_Training_Data = loadmat('__your__dataset__path') # Replace with your dataset path
    train_dataset = MatDataset(FMT_Training_Data)

   
    dm = SimpleDataModule(
        ds_train=train_dataset,
        batch_size=8, 
        num_workers=0,
        pin_memory=True
    ) 
    
    # ------------ Config --------------------
    config = get_config()
    encoder_type = config.encoder_type

    # ------------ Initialize Model ------------
    if encoder_type == 'VAE':
        model = VAE(
            in_channels=config.in_channels, 
            out_channels=config.out_channels, 
            emb_channels=config.emb_channels,
            spatial_dims=config.spatial_dims,
            hid_chs =    config.hid_chs, 
            kernel_sizes=config.kernel_sizes,
            strides =    config.strides,
            deep_supervision=config.deep_supervision,
            use_attention= config.use_attention,
            loss = torch.nn.MSELoss,
            embedding_loss_weight=config.embedding_loss_weight
        )
    elif encoder_type == 'VAEGAN':
        model = VAEGAN(
            in_channels=config.in_channels, 
            out_channels=config.out_channels, 
            emb_channels=config.emb_channels,
            spatial_dims=config.spatial_dims,
            hid_chs =    config.hid_chs,
            deep_supervision=config.deep_supervision,
            use_attention= config.use_attention,
            start_gan_train_step=-1,
            embedding_loss_weight=config.embedding_loss_weight
        )
    elif encoder_type == 'VQVAE':
        model = VQVAE(
            in_channels=config.in_channels, 
            out_channels=config.out_channels, 
            emb_channels=config.emb_channels,
            num_embeddings = config.num_embeddings,
            spatial_dims=config.spatial_dims,
            hid_chs =    config.hid_chs,
            embedding_loss_weight=1,
            beta=1,
            loss = torch.nn.L1Loss,
            deep_supervision=config.deep_supervision,
            use_attention = config.use_attention,
        )
    elif encoder_type == 'VQGAN':
        model = VQGAN(
            in_channels=config.in_channels, 
            out_channels=config.out_channels, 
            emb_channels=config.emb_channels,
            num_embeddings =  config.num_embeddings,
            spatial_dims=config.spatial_dims,
            hid_chs =    config.hid_chs,
            embedding_loss_weight=1,
            beta=1,
            start_gan_train_step=-1,
            pixel_loss = torch.nn.L1Loss,
            deep_supervision=config.deep_supervision,
            use_attention=config.use_attention,
        )
    else:
        raise ValueError(f"Invalid encoder_type: {encoder_type}")

    model.load_pretrained('__your__model__path', strict=True)
    

    # -------------- Training Initialization ---------------
    to_monitor = "train/L1"  # "val/loss" 
    min_max = "min"
    save_and_sample_every = 50

    early_stopping = EarlyStopping(
        monitor= config.es_monitor,
        min_delta= config.es_min_delta, # minimum change in the monitored quantity to qualify as an improvement
        patience= config.es_patience, # number of checks with no improvement
        mode= config.es_mode
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir), # dirpath
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=5,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator= config.accelerator,
        default_root_dir= str(path_run_dir),
        callbacks=[checkpointing],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every, 
        limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available 
        min_epochs= config.min_epochs,
        max_epochs= config.max_epochs,
        num_sanity_val_steps= config.num_sanity_val_steps,
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)


