import argparse
import torch

def get_config():
    parser = argparse.ArgumentParser(description="Configuration for the training process")
    
    # Parameters for the encoder
    parser.add_argument('--encoder_type', type=str, default='VAE',
                        choices=['VAE', 'VAEGAN', 'VQVAE', 'VQGAN'],
                        help='Type of encoder to use (VAE, VAEGAN, VQVAE, VQGAN)')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=1, help='Number of output channels')
    parser.add_argument('--emb_channels', type=int, default=8, help='Number of embedding channels')
    parser.add_argument('--spatial_dims', type=int, default=2, help='Number of spatial dimensions')
    parser.add_argument('--hid_chs', type=list, default=[64, 128, 256, 512], help='Hidden channel sizes (list)')
    parser.add_argument('--kernel_sizes', type=list, default=[3, 3, 3, 3], help='Kernel sizes (list)')
    parser.add_argument('--strides', type=list, default=[1, 2, 2, 2], help='Strides (list)')
    parser.add_argument('--deep_supervision', type=int, default=1, help='Deep supervision flag (0 or 1)')
    parser.add_argument('--use_attention', type=str, default='none', help='Attention mechanism type (string)')
    parser.add_argument('--embedding_loss_weight', type=float, default=1e-6, help='Embedding loss weight (float)')

    # Specific to VQVAE/VQGAN
    parser.add_argument('--num_embeddings', type=int, default=8192, help='Number of embeddings for VQVAE/VQGAN')

    # Early Stopping parameters
    parser.add_argument('--es_monitor', type=str, default='train/L1', help='Metric to monitor for early stopping')
    parser.add_argument('--es_min_delta', type=float, default=0.0, help='Minimum change in the monitored quantity to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=30, help='Number of checks with no improvement for early stopping')
    parser.add_argument('--es_mode', type=str, default='min', choices=['min', 'max'], help='Mode for early stopping (min or max)')

    # Trainer parameters
    parser.add_argument('--accelerator', type=str, default='gpu', help='Accelerator for training (gpu or cpu)')
    parser.add_argument('--min_epochs', type=int, default=300, help='Minimum number of epochs to train')
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of epochs to train')
    parser.add_argument('--num_sanity_val_steps', type=int, default=2, help='Number of sanity validation steps')
    parser.add_argument('--default_root_dir', type=str, default=None, help='Default root directory for trainer')
    
    # Evaluation metrics
    parser.add_argument('--metrics', type=str, nargs='+', 
                        default=['LPIPS', 'MS_SSIM', 'SSIM', 'PSNR'],
                        choices=['LPIPS', 'MS_SSIM', 'SSIM', 'PSNR', 'MSE'],
                        help='Metrics to calculate during evaluation')
    
    # Diffusion Model Parameters
    parser.add_argument('--cond_emb_dim', type=int, default=1024, help='Conditional embedding dimension')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for conditional embedding')
    parser.add_argument('--time_emb_dim', type=int, default=1024, help='Time embedding dimension')
    parser.add_argument('--unet_in_ch', type=int, default=8, help='UNet input channels')
    parser.add_argument('--unet_out_ch', type=int, default=8, help='UNet output channels')
    parser.add_argument('--unet_hid_chs', type=list, default=[64, 64, 128, 256], help='UNet hidden channel sizes (list)')
    parser.add_argument('--unet_kernel_sizes', type=list, default=[3, 3, 3, 3], help='UNet kernel sizes (list)')
    parser.add_argument('--unet_strides', type=list, default=[1, 2, 2, 2], help='UNet strides (list)')
    parser.add_argument('--unet_deep_supervision', type=bool, default=False, help='UNet deep supervision flag')
    parser.add_argument('--unet_use_res_block', type=bool, default=True, help='UNet use residual block flag')
    parser.add_argument('--unet_use_attention', type=str, default='none', help='UNet attention mechanism type (string)')
    parser.add_argument('--noise_timesteps', type=int, default=1000, help='Noise scheduler timesteps')
    parser.add_argument('--noise_beta_start', type=float, default=0.002, help='Noise scheduler beta start')
    parser.add_argument('--noise_beta_end', type=float, default=0.02, help='Noise scheduler beta end')
    parser.add_argument('--noise_schedule_strategy', type=str, default='scaled_linear', help='Noise scheduler schedule strategy (string)')

    # Sampling parameters
    parser.add_argument('--sampling_steps', type=int, default=500, help='Number of sampling steps')
    parser.add_argument('--use_ddim', type=bool, default=True, help='Use DDIM sampling instead of DDPM')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--guidance_scale', type=float, default=8.0, help='Scale for classifier-free guidance')
    parser.add_argument('--sample_size', type=tuple, default=(8, 32, 32), help='Size of generated samples (C,H,W)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--save_normalized', type=bool, default=True, help='Save normalized images')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save generated samples')

    # Dataset Sampling Parameters
    parser.add_argument('--sample_steps', type=int, nargs='+', default=[50, 150, 250], 
                        help='List of steps for sampling')
    parser.add_argument('--dataset_labels', type=dict, 
                        default={'FMT3':0, 'FMT4':1, 'FMT2':None},
                        help='Dictionary of dataset names and their labels')
    parser.add_argument('--samples_per_class', type=int, default=3000,
                        help='Number of samples to generate per class')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for sampling')
    parser.add_argument('--cfg_scale', type=float, default=4.0,
                        help='Classifier free guidance scale')
    parser.add_argument('--output_dir', type=str, 
                        default='results/FMT/samples',
                        help='Base directory for output samples')
    parser.add_argument('--img_size', type=tuple, default=(8, 8, 8),
                        help='Size of generated images (C,H,W)')
    parser.add_argument('--save_format', type=str, default='PNG',
                        choices=['PNG', 'JPG', 'JPEG'],
                        help='Format to save generated images')
    
    # Semi-supervised Learning Parameters
    parser.add_argument('--ssl_method', type=str, default='MT',
                        choices=['MT', 'FixMatch', 'MCNet'],
                        help='Semi-supervised learning method')
    parser.add_argument('--ssl_model_type', type=str, default='IPS',
                        choices=['IPS', 'EFCN', 'KNNLC'],
                        help='Base model type for SSL')
    parser.add_argument('--ssl_batch_size', type=int, default=8,
                        help='Batch size for SSL training')
    parser.add_argument('--ssl_epochs', type=int, default=100,
                        help='Number of epochs for SSL training')
    parser.add_argument('--ssl_learning_rate', type=float, default=1e-3,
                        help='Learning rate for SSL training')
    parser.add_argument('--ssl_weight_decay', type=float, default=1e-4,
                        help='Weight decay for SSL training')
    parser.add_argument('--unlabeled_path', type=str, 
                        default='DataSets/guided_input.npy',
                        help='Path to unlabeled data')
    parser.add_argument('--ssl_eval_type', type=str, default='validation',
                        help='Evaluation type for SSL')
    
    # Mean Teacher specific parameters
    parser.add_argument('--mt_ema_alpha', type=float, default=0.999,
                        help='EMA decay rate for Mean Teacher')
    parser.add_argument('--mt_consistency_weight', type=float, default=0.1,
                        help='Weight for consistency loss in Mean Teacher')
    
    # FixMatch specific parameters
    parser.add_argument('--fixmatch_confidence', type=float, default=0.95,
                        help='Confidence threshold for FixMatch pseudo-labeling')
    parser.add_argument('--fixmatch_strong_threshold', type=float, default=0.8,
                        help='Strong augmentation threshold for FixMatch')
    
    # MCNet specific parameters
    parser.add_argument('--mcnet_dropout_rate', type=float, default=0.2,
                        help='Dropout rate for MCNet')
    parser.add_argument('--mcnet_n_decoders', type=int, default=3,
                        help='Number of decoders for MCNet')
    parser.add_argument('--mcnet_consistency_weight', type=float, default=0.1,
                        help='Weight for consistency loss in MCNet')

    # Evaluation Parameters
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--eval_threshold', type=float, default=0.5,
                        help='Threshold for binary mask in evaluation metrics')
    parser.add_argument('--eval_metrics', type=list, 
                        default=['dice', 'bce', 'cnr'],
                        help='List of evaluation metrics to compute')
    parser.add_argument('--eval_save_dir', type=str,
                        default='results/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--brain_coords_path', type=str,
                        default='Tecplot_Data/BrainNodes.npy',
                        help='Path to brain coordinates file for BCE calculation')
    parser.add_argument('--eval_results_file', type=str,
                        default='evaluation_results.txt',
                        help='Name of file to save evaluation results')
    parser.add_argument('--eval_visualize', type=bool, default=False,
                        help='Whether to visualize evaluation results')
    parser.add_argument('--cnr_top_n', type=int, default=10,
                        help='Number of top pixels to use for CNR calculation')
    parser.add_argument('--cnr_bottom_n', type=int, default=50,
                        help='Number of bottom pixels to use for CNR calculation')

    return parser.parse_args()

if __name__ == '__main__':
    config = get_config()

    print("----- Encoder Parameters -----")
    print(f"Encoder Type: {config.encoder_type}")
    print(f"In Channels: {config.in_channels}")
    print(f"Out Channels: {config.out_channels}")
    print(f"Embedding Channels: {config.emb_channels}")
    print(f"Spatial Dimensions: {config.spatial_dims}")
    print(f"Hidden Channels: {config.hid_chs}")
    print(f"Kernel Sizes: {config.kernel_sizes}")
    print(f"Strides: {config.strides}")
    print(f"Deep Supervision: {config.deep_supervision}")
    print(f"Use Attention: {config.use_attention}")
    print(f"Embedding Loss Weight: {config.embedding_loss_weight}")
    if hasattr(config, 'num_embeddings'):
        print(f"Number of Embeddings: {config.num_embeddings}")

    print("\n----- Early Stopping Parameters -----")
    print(f"Early Stopping Monitor: {config.es_monitor}")
    print(f"Early Stopping Min Delta: {config.es_min_delta}")
    print(f"Early Stopping Patience: {config.es_patience}")
    print(f"Early Stopping Mode: {config.es_mode}")

    print("\n----- Trainer Parameters -----")
    print(f"Accelerator: {config.accelerator}")
    print(f"Min Epochs: {config.min_epochs}")
    print(f"Max Epochs: {config.max_epochs}")
    print(f"Num Sanity Val Steps: {config.num_sanity_val_steps}")
    print(f"Default Root Dir: {config.default_root_dir}")
    
    print("\n----- Evaluation Metrics -----")
    print(f"Metrics: {config.metrics}")
    
    print("\n----- Diffusion Model Parameters -----")
    print(f"Conditional Embedding Dimension: {config.cond_emb_dim}")
    print(f"Number of Classes: {config.num_classes}")
    print(f"Time Embedding Dimension: {config.time_emb_dim}")
    print(f"UNet Input Channels: {config.unet_in_ch}")
    print(f"UNet Output Channels: {config.unet_out_ch}")
    print(f"UNet Hidden Channels: {config.unet_hid_chs}")
    print(f"UNet Kernel Sizes: {config.unet_kernel_sizes}")
    print(f"UNet Strides: {config.unet_strides}")
    print(f"UNet Deep Supervision: {config.unet_deep_supervision}")
    print(f"UNet Use Residual Block: {config.unet_use_res_block}")
    print(f"UNet Use Attention: {config.unet_use_attention}")
    print(f"Noise Timesteps: {config.noise_timesteps}")
    print(f"Noise Beta Start: {config.noise_beta_start}")
    print(f"Noise Beta End: {config.noise_beta_end}")
    print(f"Noise Schedule Strategy: {config.noise_schedule_strategy}")

    print("\n----- Sampling Parameters -----")
    print(f"Sampling Steps: {config.sampling_steps}")
    print(f"Use DDIM: {config.use_ddim}")
    print(f"Number of Samples: {config.n_samples}")
    print(f"Guidance Scale: {config.guidance_scale}")
    print(f"Sample Size: {config.sample_size}")
    print(f"Seed: {config.seed}")
    print(f"Save Normalized: {config.save_normalized}")
    print(f"Results Directory: {config.results_dir}")

    print("\n----- Dataset Sampling Parameters -----")
    print(f"Sample Steps: {config.sample_steps}")
    print(f"Dataset Labels: {config.dataset_labels}")
    print(f"Samples Per Class: {config.samples_per_class}")
    print(f"Batch Size: {config.batch_size}")
    print(f"CFG Scale: {config.cfg_scale}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Image Size: {config.img_size}")
    print(f"Save Format: {config.save_format}")

    print("\n----- Semi-supervised Learning Parameters -----")
    print(f"SSL Method: {config.ssl_method}")
    print(f"SSL Model Type: {config.ssl_model_type}")
    print(f"SSL Batch Size: {config.ssl_batch_size}")
    print(f"SSL Epochs: {config.ssl_epochs}")
    print(f"SSL Learning Rate: {config.ssl_learning_rate}")
    print(f"SSL Weight Decay: {config.ssl_weight_decay}")
    print(f"Unlabeled Path: {config.unlabeled_path}")
    print(f"SSL Eval Type: {config.ssl_eval_type}")

    print("\n----- Mean Teacher Parameters -----")
    print(f"MT EMA Alpha: {config.mt_ema_alpha}")
    print(f"MT Consistency Weight: {config.mt_consistency_weight}")

    print("\n----- FixMatch Parameters -----")
    print(f"FixMatch Confidence: {config.fixmatch_confidence}")
    print(f"FixMatch Strong Threshold: {config.fixmatch_strong_threshold}")

    print("\n----- MCNet Parameters -----")
    print(f"MCNet Dropout Rate: {config.mcnet_dropout_rate}")
    print(f"MCNet Number of Decoders: {config.mcnet_n_decoders}")
    print(f"MCNet Consistency Weight: {config.mcnet_consistency_weight}")

    print("\n----- Evaluation Parameters -----")
    print(f"Eval Batch Size: {config.eval_batch_size}")
    print(f"Eval Threshold: {config.eval_threshold}")
    print(f"Eval Metrics: {config.eval_metrics}")
    print(f"Eval Save Dir: {config.eval_save_dir}")
    print(f"Brain Coords Path: {config.brain_coords_path}")
    print(f"Eval Results File: {config.eval_results_file}")
    print(f"Eval Visualize: {config.eval_visualize}")
    print(f"CNR Top N: {config.cnr_top_n}")
    print(f"CNR Bottom N: {config.cnr_bottom_n}")
