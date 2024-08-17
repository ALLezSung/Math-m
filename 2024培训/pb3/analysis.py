import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 读取CSV文件
df = pd.read_csv(r'pb3/2016-2023.csv')
df = df[72:]

# 提取数据
data_water = df['水'].values
data_sand = df['沙'].values

# # 延申数据
# extend_length = int(len(df) / 2)  # 延申长度，转换为整数
# data_water_extended = np.pad(data_water, (extend_length, extend_length), 'reflect')
# data_sand_extended = np.pad(data_sand, (extend_length, extend_length), 'reflect')

# 进行小波变换
coeffs_water = pywt.wavedec(data_water, 'db1', level=5)
coeffs_sand = pywt.wavedec(data_sand, 'db1', level=5)

# # 去掉延申部分
# coeffs_water = [c[extend_length:-extend_length] for c in coeffs_water]
# coeffs_sand = [c[extend_length:-extend_length] for c in coeffs_sand]

# 将小波系数转换为二维数组
coeffs_water_2d = np.array([np.pad(c, (0, len(data_water) - len(c)), 'constant') for c in coeffs_water])
coeffs_sand_2d = np.array([np.pad(c, (0, len(data_sand) - len(c)), 'constant') for c in coeffs_sand])

print(coeffs_water_2d, coeffs_sand_2d)

# 绘制小波系数等值线图
plt.figure(figsize=(12, 6))

# 水的小波系数等值线图
plt.subplot(2, 2, 1)
plt.contourf(coeffs_water_2d, cmap='viridis')
plt.title('水的小波系数等值线图')
plt.colorbar()

# 沙的小波系数等值线图
plt.subplot(2, 2, 2)
plt.contourf(coeffs_sand_2d, cmap='viridis')
plt.title('沙的小波系数等值线图')
plt.colorbar()

# 计算小波方差
variance_water = [np.var(c) for c in coeffs_water]
variance_sand = [np.var(c) for c in coeffs_sand]

# 绘制小波方差图
plt.subplot(2, 2, 3)
plt.plot(variance_water, marker='o')
plt.title('水的小波方差图')
plt.xlabel('尺度')
plt.ylabel('方差')

plt.subplot(2, 2, 4)
plt.plot(variance_sand, marker='o')
plt.title('沙的小波方差图')
plt.xlabel('尺度')
plt.ylabel('方差')

plt.tight_layout()
plt.show()