import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib_inline import backend_inline 
backend_inline.set_matplotlib_formats('svg')
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  


# 设置全局变量
SPEED_OF_SOUND = 0.34  # 声速 km/s
X_LENGTH = 97.304    # 单位经度长度 km
Y_LENGTH = 111.263   # 单位纬度长度 km

# 导入初始数据
DATA_LIST = np.array([
    [110.241, 27.204, 824, 100.767],
    [110.780, 27.456, 727, 112.220],
    [110.712, 27.785, 742, 188.020],
    [110.251, 27.825, 850, 258.985],
    [110.524, 27.617, 786, 118.443],
    [110.467, 27.921, 678, 266.871],
    [110.047, 27.121, 575, 163.024]
])
DATA_DF = pd.DataFrame(DATA_LIST, columns=['longitude', 'latitude', 'altitude', 'time_received'])

# 数据预处理
DATA_DF['x'] = DATA_DF['longitude'] * X_LENGTH
DATA_DF['y'] = DATA_DF['latitude'] * Y_LENGTH
DATA_DF['z'] = DATA_DF['altitude'] / 1000

# 计算位置和优化目标函数
def calculate_position(df, tol=1e-9):
    points = df[['x', 'y', 'z']].to_numpy()
    times_received = df['time_received'].to_numpy()
    
    def objective_function(variables, points, times_received):
            signal_origin = variables[:3]
            signal_time = variables[3]
            euclid_distances = np.linalg.norm(points - signal_origin, axis=1)
            sound_distances = (times_received - signal_time) * SPEED_OF_SOUND
            return np.mean(np.abs(euclid_distances - sound_distances))
    
    # 限制函数与边界条件
    def constraint_function(variables, times_received):
        signal_time = variables[3]
        return times_received - signal_time
    constraints = {'type': 'ineq', 'fun':constraint_function}
    bounds = [(-180*97.304, 180*97.304), (-90*111.263, 90*111.263), (0, 10000), (0, np.min(times_received))]
    
    # 初始猜测
    x0 = np.zeros(4)
    x0[:3] = np.mean(points, axis=0)  # 假设初始位置为所有监测设备位置的平均值
    x0[3] = 0  # 假设初始信号发出时间为0

    # 实施优化
    result = minimize(objective_function, x0, 
                      args=(points, times_received), 
                      method='L-BFGS-B', options={'gtol': tol},
                      constraints=constraints, bounds=bounds
                      )
    
    if result.success:
        RESULT_POINT = pd.DataFrame({'longitude': result.x[0]/97.304, 'latitude': result.x[1]/111.263, 'altitude': result.x[2]*1000, 
                                     'time_explosion': result.x[3], 'x':result.x[0], 'y':result.x[1], 'z':result.x[2]}, index=['Position'])
        return RESULT_POINT  # 返回信号源位置
    else:
        raise ValueError("Optimization failed: " + result.message)


# 误差函数
def calculate_errors(df, target_position):
    errors = []
    
    target_xyz = np.array(calculated_position.loc['Position'].to_list()[-3:])
    target_t = calculated_position.loc['Position','time_explosion']
    for index, row in df.iterrows():
        error = np.sqrt(np.sum((row[['x', 'y', 'z']] - target_xyz) ** 2)) - (row['time_received'] - target_t) * SPEED_OF_SOUND
        errors.append(error)
    errors_abs = np.abs(errors)

    return errors, np.mean(errors_abs), np.std(errors_abs)

# 绘图函数
def plot_devices_and_target(df, target_position):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取设备坐标点
    xs = df['longitude']
    ys = df['latitude']
    zs = df['altitude']

    # 绘制设备点
    ax.scatter(xs, ys, zs, c='blue', label='设备点', depthshade=True)

    # 绘制信号源点
    target_position = target_position.loc['Position'].to_dict()
    ax.scatter(target_position['longitude'], target_position['latitude'], target_position['altitude'], c='red', marker='*', label='预测目标点')

    # 设置图例
    ax.legend()
    ax.set_xlabel('经 度（°）')
    ax.set_ylabel('维 度（°）')
    ax.set_zlabel('高 程（m）')

    plt.show()


if __name__ == '__main__':

    # 调用函数进行计算
    calculated_position = calculate_position(DATA_DF)
    print(f'监测设备信息:\n{DATA_DF}\n预测目标点信息:\n{calculated_position}')
    print(calculate_errors(DATA_DF, calculated_position))

    # 调用函数进行绘图
    plot_devices_and_target(DATA_DF, calculated_position)
