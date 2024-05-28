import numpy as np
from scipy.optimize import least_squares

# 已知数据（加入误差前）
coords = np.array([
    [110.241, 27.204, 824],
    [110.783, 27.456, 727],
    [110.762, 27.785, 742],
    [110.251, 28.025, 850],
    [110.524, 27.617, 786],
    [110.467, 28.081, 678],
    [110.047, 27.521, 575]
])

times = np.array([
    [100.767, 164.229, 214.850, 270.065],
    [92.453, 112.220, 169.362, 196.583],
    [75.560, 110.696, 156.936, 188.020],
    [94.653, 141.409, 196.517, 258.985],
    [78.600, 86.216, 118.443, 126.669],
    [67.274, 166.270, 175.482, 266.871],
    [103.738, 163.024, 206.789, 210.306]
])

# 震动波传播速度
v = 340

# 转换经纬度为距离
lat_dist = 111.263
lon_dist = 97.304
coords[:, 0] *= lon_dist
coords[:, 1] *= lat_dist

# 加入时间误差
np.random.seed(42)  # 设置随机种子以获得可重复结果
time_errors = np.random.normal(0, 0.5, times.shape)
times_with_error = times + time_errors

# 构建误差函数
def residuals_with_error(params):
    positions = params[:4*3].reshape(4, 3)
    times0 = params[4*3:]
    residuals = []
    for k in range(4):  # 对每个残骸
        x0, y0, z0, t0 = positions[k, 0], positions[k, 1], positions[k, 2], times0[k]
        pred_times = np.sqrt((coords[:, 0] - x0)**2 + (coords[:, 1] - y0)**2 + (coords[:, 2] - z0)**2) / v + t0
        residuals.extend(pred_times - times_with_error[:, k])
    return np.array(residuals)

# 初始猜测值
initial_guess = np.hstack((np.tile([coords[:, 0].mean(), coords[:, 1].mean(), coords[:, 2].mean()], 4), times_with_error.mean(axis=0)))

# 使用最小二乘法求解
result = least_squares(residuals_with_error, initial_guess)
positions_with_error, times0_with_error = result.x[:4*3].reshape(4, 3), result.x[4*3:]

# 转换回经纬度
positions_with_error[:, 0] /= lon_dist
positions_with_error[:, 1] /= lat_dist

for i in range(4):
    print(f"修正后残骸{i+1} 音爆发生位置：经度={positions_with_error[i, 0]}, 纬度={positions_with_error[i, 1]}, 高程={positions_with_error[i, 2]}, 时间={times0_with_error[i]}")
