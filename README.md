# 📘 DANG - Data Augmentation based on NIR-II Guidance for Fluorescence Molecular Tomography

## Paper

![](media/DANG.png)
*Figure: Framework for DANG strategy*

![](media/NIR-II_Synthesis.png)
*Figure: Synthetic results for NIR-II Guided Diffusion model*

This project provides the source code for the paper on Data Augmentation based on NIR-II Guidance for Fluorescence Molecular Tomography. The paper introduces a novel approach to enhance the quality and diversity of training data for fluorescence molecular tomography by leveraging NIR-II guidance. More details can be found in the paper.

## Contributions

*   We proposed a novel data augmentation based on NIR-II Guided diffusion model (DANG) for FMT tasks, which introduced two key innovations: on the one hand, this is the first framework to integrate a diffusion model as a data augmentation tool generating high-fidelity, diverse NIR-II fluorescence samples to overcome pattern diversity limitations in existing datasets. On the other hand, a label-free utilization of synthetic data is carried out by semi-supervised learning, enabling reconstruction networks to leverage unlabeled synthetic samples.
*  A NIR-II specific guidance mechanism is introduced, leveraging the analogy between photon propagation in Monte Carlo simulations and the diffusion sampling process, which is used to control the generating of NIR-II fluorescence samples. Quantitative evaluations demonstrate that the guidance filled the gaps in feature space of real
fluorescence samples, outperforming existing diffusion models in both fidelity and diversity.
*  To harness unlabeled synthetic samples, we adopt semi-supervised training strategies across three distinct reconstruction networks. Experiments show consistent performance gains in both simulation and in vivo experiments, with improved localization accuracy and robustness to multi-target reconstruction, surpassing fully supervised baselines.


## 🛠️ Prerequisites

*   🐍 Python 3.8+
*   🧬 Git
*   🔑 Install the required Python packages as following instruction


## 📂 File Structure

*   `config.py`: Configuration file containing all adjustable parameters for training and evaluation.
*   `Label_Free_Training.py`: Main script for semi-supervised learning with Mean Teacher, FixMatch and MCNet methods.
*   `sample_dataset.py`: Script for generating synthetic NIR-II fluorescence samples using trained diffusion model.
*   `train_diffusion.py`: Script for training the NIR-II guided diffusion model.
*   `train_encoder.py`: Script for training the latent embedder (VAE, VAEGAN, VQVAE, VQGAN).
*   `evaluate_images.py`: Script for evaluating generated synthetic images quality.
*   `evaluate_encoder.py`: Script for evaluating the latent embedder performance.

### 📁 FMT/
*   `FMT_Models/`: Implementation of various FMT reconstruction networks (IPS, EFCN, KNNLC).
*   `FMT_dataset.py`: Dataset loading and processing utilities for FMT data.
*   `SSL_Algorithms.py`: Implementation of semi-supervised learning algorithms.

### 📁 NIR_II_Guided_Diffusion/
*   `models/`: Core model architectures for diffusion and guidance.
*   `utils/`: Utility functions for training and evaluation.
*   `pipelines.py`: End-to-end pipelines for model training and inference.

*   `requirements.txt`: Python dependencies.
*   `README.md`: Project documentation and instructions.


## Dataset Information

### 🔒 Data Availability
Due to licensing and ethical constraints, we cannot publicly distribute the full training dataset. However, we provide the following resources to support your work:

### 📁 Provided Resources
| Resource | Description | Access |
|----------|-------------|--------|
| Sample Dataset | 200 representative training samples (10% of full dataset) | [Download Link](#) |
| Pre-trained Models | Our best-performing DANG model weights | [Model Zoo](#) |
| Synthetic Data Generator | Python scripts to create similar training data | `/scripts/data_generation` |


## Model Implementation

### 🧠 NIR-II Guided Data-Augmentation
This repository contains:
- Source code for our diffusion-based data augmentation framework
- Pre-trained models for NIR-II fluorescence sample generation

### 🔍 FMT Reconstruction Networks
For reconstruction network implementations, please refer to our previous works:

1. **Excitation-based FCN**  
   ```bibtex
   @article{cao2022excitation,
     title={Excitation-based fully connected network for precise NIR-II fluorescence molecular tomography},
     author={Cao, Chen and others},
     journal={Biomed. Opt. Express},
     volume={13},
     number={12},
     pages={6284},
     year={2022}
   }
2. **Positional Encoding (TSPE)**  
   ```bibtex
   @article{han2025tspe,
    title={TSPE: Reconstruction of multi-morphological tumors of NIR-II fluorescence molecular tomography based on positional encoding},
    author={Han, Kai and others},
    journal={Computer Methods and Programs in Biomedicine},
    volume={261},
    pages={108554},
    year={2025}
   }
3. **IPS**  
   ```bibtex
   @article{gao2018nonmodel,
    title={Nonmodel-based bioluminescence tomography using a machine-learning reconstruction strategy},
    author={Gao, Yuan and others},
    journal={Optica},
    volume={5},
    number={11},
    pages={1451},
    year={2018}
   }


## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

### 🚀 Quick Start

1.  **Download sample data**

    ```bash
    wget https://example.com/data/sample_dataset.zip
    unzip sample_dataset.zip
    ```

2.  **Train encoder**

    *   Execute the script for training the encoder with specified arguments. You can choose from VAE, VAEGAN, VQVAE, or VQGAN. Here's an example for training a VAE:

    ```bash
    python train_encoder.py \
      --encoder_type VAE \
      --in_channels 1 \
      --out_channels 1 \
      --emb_channels 8 \
      --spatial_dims 2 \
      --hid_chs 64 128 256 512 \
      --kernel_sizes 3 3 3 3 \
      --strides 1 2 2 2 \
      --deep_supervision 1 \
      --use_attention none \
      --loss torch.nn.MSELoss \
      --embedding_loss_weight 1e-6
    ```

3.  **Validate encoder**

    *   Validate the performance of the trained encoder. Make sure to specify the correct `--encoder_type` and provide the path to the encoder checkpoint. You can also specify the metrics to be calculated using the `--metrics` argument.

    ```bash
    python evaluate_encoder.py \
      --encoder_type VAE \
      --in_channels 1 \
      --out_channels 1 \
      --emb_channels 8 \
      --spatial_dims 2 \
      --hid_chs 64 128 256 512 \
      --kernel_sizes 3 3 3 3 \
      --strides 1 2 2 2 \
      --deep_supervision 1 \
      --use_attention none \
      --loss torch.nn.MSELoss \
      --embedding_loss_weight 1e-6 \
      --metrics LPIPS MS_SSIM SSIM PSNR
    ```

4.  **Train Diffusion Model**

    *   Train the diffusion model using the training script. You can customize various parameters like embedding dimensions, UNet architecture, and noise scheduler settings:

    ```bash
    python train_diffusion.py \
      --cond_emb_dim 1024 \
      --num_classes 2 \
      --time_emb_dim 1024 \
      --unet_in_ch 8 \
      --unet_out_ch 8 \
      --unet_hid_chs 64 64 128 256 \
      --unet_kernel_sizes 3 3 3 3 \
      --unet_strides 1 2 2 2 \
      --unet_deep_supervision False \
      --unet_use_res_block True \
      --unet_use_attention none \
      --noise_timesteps 1000 \
      --noise_beta_start 0.002 \
      --noise_beta_end 0.02 \
      --noise_schedule_strategy scaled_linear
    ```

    *   For a simpler training setup with default parameters:

    ```bash
    python train_diffusion.py
    ```

    *   For GPU-specific training with custom epochs:

    ```bash
    python train_diffusion.py \
      --accelerator gpu \
      --min_epochs 300 \
      --max_epochs 500
    ```

5.  **Sampling NIR-II samples and evaluate synthetic results**

    *   Generate synthetic NIR-II fluorescence samples using the trained diffusion model. You can customize various sampling parameters:

    ```bash
    python sample_dataset.py \
      --sample_steps 50 150 250 \
      --samples_per_class 3000 \
      --batch_size 100 \
      --cfg_scale 4.0 \
      --output_dir results/FMT/samples \
      --img_size 8 8 8 \
      --save_format PNG \
      --seed 0
    ```

    *   For quick testing with fewer samples:

    ```bash
    python sample_dataset.py \
      --sample_steps 50 \
      --samples_per_class 100 \
      --batch_size 20
    ```

    *   For generating high-quality samples with more guidance:

    ```bash
    python sample_dataset.py \
      --sample_steps 250 \
      --samples_per_class 5000 \
      --cfg_scale 7.5 \
      --batch_size 50 \
      --seed 42
    ```

    *   The generated samples will be saved in the specified output directory with the following structure:
    ```
    results/
    └── FMT/
        └── samples/
            └── generated_diffusion3_[steps]/
                ├── FMT2/
                ├── FMT3/
                └── FMT4/
    ```
6.  **Implement Label-free FMT Neural Network Training**

    *   Train FMT reconstruction networks using semi-supervised learning with synthetic data. You can choose from different SSL methods (Mean Teacher, FixMatch, MCNet):

    ```bash
    python Label_Free_Training.py \
      --ssl_method MT \
      --ssl_model_type IPS \
      --ssl_batch_size 16 \
      --ssl_epochs 200 \
      --mt_ema_alpha 0.999 \
      --mt_consistency_weight 0.1 \
      --ssl_learning_rate 1e-3 \
      --unlabeled_path DataSets/guided_input.npy
    ```

    *   For FixMatch-based training:

    ```bash
    python Label_Free_Training.py \
      --ssl_method FixMatch \
      --ssl_model_type EFCN \
      --ssl_batch_size 32 \
      --fixmatch_confidence 0.95 \
      --ssl_learning_rate 1e-3 \
      --ssl_weight_decay 1e-4
    ```

    *   For MCNet-based training with multiple decoders:

    ```bash
    python Label_Free_Training.py \
      --ssl_method MCNet \
      --ssl_model_type KNNLC \
      --mcnet_dropout_rate 0.2 \
      --mcnet_n_decoders 3 \
      --mcnet_consistency_weight 0.1 \
      --ssl_batch_size 16 \
      --ssl_epochs 300
    ```

    *   The trained models will be saved in the following structure:
    ```
    checkpoints/
    └── ssl_models/
        ├── IPS_MT.pth
        ├── EFCN_FixMatch.pth
        └── KNNLC_MCNet.pth
    ```



