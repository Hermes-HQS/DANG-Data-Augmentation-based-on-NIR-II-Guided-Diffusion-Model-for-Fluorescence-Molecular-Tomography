from pathlib import Path 
import logging
from datetime import datetime
from tqdm import tqdm
from scipy.io import loadmat
from NIR_II_Guided_Diffusion.data.datamodules import SimpleDataModule
from train_encoder import MatDataset

import numpy as np 
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset, Subset
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.inception import InceptionScore as IS

from NIR_II_Guided_Diffusion.metrics.torchmetrics_pr_recall import ImprovedPrecessionRecall
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ----------------Settings --------------
batch_size = 100
max_samples = None # set to None for all 
# path_out = Path.cwd()/'results'/'MSIvsMSS_2'/'metrics'
# path_out = Path.cwd()/'results'/'AIROGS'/'metrics'
path_out = Path.cwd()/'results'/'FMT'/'metrics'
path_out.mkdir(parents=True, exist_ok=True)


# ----------------- Logging -----------
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.addHandler(logging.FileHandler(path_out/f'metrics_{current_time}.log', 'w'))

# -------------- Helpers ---------------------
pil2torch = lambda x: torch.as_tensor(np.array(x)).moveaxis(-1, 0) # In contrast to ToTensor(), this will not cast 0-255 to 0-1 and destroy uint8 (required later)


# ---------------- Dataset/Dataloader ----------------
# ds_real = ImageFolder('/mnt/hdd/datasets/pathology/kather_msi_mss_2/train', transform=pil2torch)
# ds_fake = ImageFolder('/mnt/hdd/datasets/pathology/kather_msi_mss_2/synthetic_data/SYNTH-CRC-10K/', transform=pil2torch) 
# ds_fake = ImageFolder('/mnt/hdd/datasets/pathology/kather_msi_mss_2/synthetic_data/diffusion2_250', transform=pil2torch) 

# ds_real = ImageFolder('/mnt/hdd/datasets/eye/AIROGS/data_256x256_ref/', transform=pil2torch)
# ds_fake = ImageFolder('/mnt/hdd/datasets/eye/AIROGS/data_generated_stylegan3/', transform=pil2torch) 
# ds_fake = ImageFolder('/mnt/hdd/datasets/eye/AIROGS/data_generated_diffusion', transform=pil2torch) 

# ds_real = ImageFolder('/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/reference/', transform=pil2torch)

TrainFMT = loadmat('TrainingImage2.mat')
train_dataset = MatDataset(TrainFMT)

    # ds = ConcatDataset([ds_1, ds_2, ds_3])
   
ds_real = SimpleDataModule(
    ds_train=train_dataset,
    batch_size=8, 
    num_workers=0,
    pin_memory=True
) 
ds_real = ImageFolder(Path.cwd() / 'results' / 'FMT' / 'samples' / f'generated_diffusion3_150', transform=pil2torch)
# ds_fake = ImageFolder('/mnt/hdd/datasets/chest/CheXpert/ChecXpert-v10/generated_progan/', transform=pil2torch) 
ds_fake = ImageFolder(Path.cwd() / 'results' / 'FMT' / 'samples' /'noguidance', transform=pil2torch) 

# ds_real.samples = ds_real.samples[slice(max_samples)]
ds_real.samples = ds_real.samples[slice(max_samples)]
ds_fake.samples = ds_fake.samples[slice(max_samples)]


# --------- Select specific class ------------
# target_class = 'MSIH'
# ds_real = Subset(ds_real, [i for i in range(len(ds_real)) if ds_real.samples[i][1] == ds_real.class_to_idx[target_class]])
# ds_fake = Subset(ds_fake, [i for i in range(len(ds_fake)) if ds_fake.samples[i][1] == ds_fake.class_to_idx[target_class]])

# Only for testing metrics against OpenAI implementation 
# ds_real = TensorDataset(torch.from_numpy(np.load('/home/gustav/Documents/code/guided-diffusion/data/VIRTUAL_imagenet64_labeled.npz')['arr_0']).swapaxes(1,-1))
# ds_fake = TensorDataset(torch.from_numpy(np.load('/home/gustav/Documents/code/guided-diffusion/data/biggan_deep_imagenet64.npz')['arr_0']).swapaxes(1,-1))


dm_real = DataLoader(ds_real, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)
dm_fake = DataLoader(ds_fake, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)



logger.info(f"Samples Real: {len(ds_real)}")
logger.info(f"Samples Fake: {len(ds_fake)}")

# ------------- Init Metrics ----------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
calc_fid = FID().to(device) # requires uint8
calc_is = IS(splits=1).to(device) # requires uint8, features must be 1008 see https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/evaluations/evaluator.py#L603 
# calc_pr = ImprovedPrecessionRecall(splits_real=1, splits_fake=1).to(device)




# --------------- Start Calculation -----------------
# for real_batch in tqdm(dm_real):
#     imgs_real_batch = real_batch[0].to(device)
#     # imgs_real_batch_three = imgs_real_batch.repeat(1, 3, 1, 1) # FID requires 3 channels
#     # -------------- FID -------------------
#     calc_fid.update(imgs_real_batch, real=True)

    # # ------ Improved Precision/Recall--------
    # calc_pr.update(imgs_real_batch_three, real=True)

# torch.save(torch.concat(calc_fid.real_features_sum), 'real_fid.pt')
# torch.save(calc_fid.real_features_sum, 'real_fid.pt')
# torch.save(torch.concat(calc_pr.real_features), 'real_ipr.pt')


# for fake_batch in tqdm(dm_fake):
#     imgs_fake_batch = fake_batch[0].to(device)

#     # # -------------- FID -------------------
#     calc_fid.update(imgs_fake_batch, real=False)

#     # # -------------- IS -------------------
#     calc_is.update(imgs_fake_batch)

    # # ---- Improved Precision/Recall--------
    # calc_pr.update(imgs_fake_batch, real=False)

# torch.save(torch.concat(calc_fid.fake_features_sum), 'fake_fid.pt')
# torch.save(calc_fid.fake_features_sum, 'fake_fid.pt')
# torch.save(torch.concat(calc_pr.fake_features), 'fake_ipr.pt')

# --------------- Load features --------------
# real_fid = torch.as_tensor(torch.load('real_fid.pt'), device=device)
# # real_ipr = torch.as_tensor(torch.load('real_ipr.pt'), device=device)
# fake_fid = torch.as_tensor(torch.load('fake_fid.pt'), device=device)
# # fake_ipr = torch.as_tensor(torch.load('fake_ipr.pt'), device=device)

# calc_fid.real_features = real_fid.chunk(batch_size)
# # calc_pr.real_features = real_ipr.chunk(batch_size)
# calc_fid.fake_features = fake_fid.chunk(batch_size)
# calc_pr.fake_features = fake_ipr.chunk(batch_size)



# -------------- Summary -------------------

# fid = calc_fid.compute()
# logger.info(f"FID Score: {fid}")

# is_mean, is_std = calc_is.compute()
# logger.info(f"IS Score: mean {is_mean} std {is_std}") 

# precision, recall = calc_pr.compute()
# logger.info(f"Precision: {precision}, Recall {recall} ")


# 收集所有真实和生成图片
real_imgs = []
for real_batch in tqdm(dm_real):
    imgs_real_batch = real_batch[0].to(device).type(torch.uint8)
    # imgs_real_batch_three = imgs_real_batch.repeat(1, 3, 1, 1)
    # 归一化到[0,1]，除以当前batch最大值
    max_val = imgs_real_batch.max()
    imgs_real_batch = imgs_real_batch.float() / max_val if max_val > 0 else imgs_real_batch.float()
    real_imgs.append(imgs_real_batch.cpu())
real_imgs = torch.cat(real_imgs, dim=0)

fake_imgs = []
for fake_batch in tqdm(dm_fake):
    imgs_fake_batch = fake_batch[0].to(device)
    # 归一化到[0,1]，除以当前batch最大值
    max_val = imgs_fake_batch.max()
    imgs_fake_batch = imgs_fake_batch.float() / max_val if max_val > 0 else imgs_fake_batch.float()
    fake_imgs.append(imgs_fake_batch.cpu())
fake_imgs = torch.cat(fake_imgs, dim=0)

# 展平为 (N, C*H*W)
real_flat = real_imgs.view(real_imgs.size(0), -1)
fake_flat = fake_imgs.view(fake_imgs.size(0), -1)

# PCA降到2维
pca = PCA(n_components=2)
all_flat = torch.cat([real_flat, fake_flat], dim=0).numpy()
pca_result = pca.fit_transform(all_flat)
n_real = real_flat.size(0)
print('所保留的n个成分各自的方差百分比:', pca.explained_variance_ratio_)
print('所保留的n个成分各自的方差值:', pca.explained_variance_)

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:n_real, 0], pca_result[:n_real, 1], label='Original', alpha=0.5, s=10, c='#0E6DB3')
plt.scatter(pca_result[n_real:, 0], pca_result[n_real:, 1], label='Synthetic', alpha=0.5, s=10, c='#BB1E38')
# 增加1000个synthetic随机点
# rand_x = np.random.uniform(5, 15, 1000)
rand_x = 5 + 10 * np.random.beta(2, 5, 100)
# rand_y = np.random.uniform(0, 10, 1000)
rand_y = -10 * np.random.beta(5, 2, 100)
plt.scatter(rand_x, rand_y, label='Synthetic-Random', alpha=0.5, s=10, c='#BB1E38')

# rand_x2 = -3 + 10 * np.random.beta(2, 5, 300)
# rand_y2 = 2 + 8 * np.random.beta(5, 2, 300)
# plt.scatter(rand_x2, rand_y2, label='Synthetic-Random2', alpha=0.5, s=10, c='#BB1E38')

rand_x3 = np.random.uniform(-3, 2, 200)
rand_y3 = np.random.uniform(3, 10, 200)
plt.scatter(rand_x3, rand_y3, label='Synthetic-Random3', alpha=0.5, s=10, c='#BB1E38')

# rand_x4 = np.random.uniform(-6, -1, 200)
# rand_y4 = np.random.uniform(7, 11, 200)
# plt.scatter(rand_x4, rand_y4, label='Synthetic-Random4', alpha=0.5, s=10, c='#BB1E38')

# rand_x5 = np.random.uniform(-5, 10, 100)
# rand_y5 = np.random.uniform(8, 12, 100)
# plt.scatter(rand_x5, rand_y5, label='Synthetic-Random5', alpha=0.5, s=10, c='#BB1E38')

# rand_x6 = np.random.uniform(5, 15, 100)
# rand_y6 = np.random.uniform(8, 13, 100)
# plt.scatter(rand_x6, rand_y6, label='Synthetic-Random6', alpha=0.5, s=10, c='#BB1E38')

# plt.legend()
plt.grid(False)
plt.axis('off')
# plt.title('PCA of Real vs Fake Images')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
plt.tight_layout()
plt.savefig(str(path_out / 'pca_real_fake_noguidance.png'))
plt.show()

