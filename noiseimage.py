import numpy as np
import matplotlib.pyplot as plt

# 创建256x256全零图像
img = np.zeros((256, 256), dtype=np.float32)

# 椭圆参数（适当增大，保证平滑）
center_x, center_y = 172, 83   # 中间偏右
axis_x, axis_y = 20, 17         # 椭圆长短轴（半径）
max_value = 1                 # 椭圆中心像素值

# 填充带梯度的椭圆
Y, X = np.ogrid[:256, :256]
dist = ((X - center_x) / axis_x) ** 2 + ((Y - center_y) / axis_y) ** 2
mask = dist <= 1
img[mask] = max_value * (1 - dist[mask])

# 在椭圆周围加噪声（距离在1~1.3之间的环形区域）
noise_ring = (dist > 0.7) & (dist <= 1.3)
noise = np.random.uniform(0, 0.16, size=img.shape)
img[noise_ring] = img[noise_ring] + noise[noise_ring]

# 对整张图像加随机噪声（例如均匀噪声，幅度可调）
global_noise = np.random.uniform(0, 0.45, size=img.shape)
img = img + global_noise

# 可视化
plt.imshow(img, cmap='jet')
# plt.colorbar()
# plt.title('Gradient Ellipse with Noisy Ring and Global Noise in 256x256 Image')
plt.axis('off')  # 关闭坐标轴
plt.show()