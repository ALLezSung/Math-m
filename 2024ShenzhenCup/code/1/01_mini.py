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
errors = []
def calculate_position(df, tol=1e-9):
    points = df[['x', 'y', 'z']].to_numpy()
    times_received = df['time_received'].to_numpy()
    
    def objective_function(variables, points, times_received):
            requ_1 = np.concatenate((points[0, :], times_received[0].reshape(-1)))
            signal_origin = variables[:3]
            signal_time = variables[3]

            xleft = 2*(requ_1[0] - points[:, 0]) * signal_origin[0]
            yleft = 2*(requ_1[1] - points[:, 1]) * signal_origin[1]
            zleft = 2*(requ_1[2] - points[:, 2]) * signal_origin[2]
            tleft = 2*(requ_1[3] - times_received) * (SPEED_OF_SOUND **2) * signal_time

            vright = SPEED_OF_SOUND ** 2 * (signal_time ** 2 - np.full(signal_time.shape, requ_1[3]) ** 2)
            xyzright = np.sum(requ_1[0:3] ** 2 - signal_origin ** 2)

            error = abs(xleft + yleft + zleft + tleft - vright - xyzright)

            return np.sum(error)
    
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
    

def calculate_errors(df, target_position):
    errors = []
    
    target_xyz = np.array(calculated_position.loc['Position'].to_list()[-3:])
    target_t = calculated_position.loc['Position','time_explosion']
    for index, row in df.iterrows():
        error = np.sqrt(np.sum((row[['x', 'y', 'z']] - target_xyz) ** 2)) - (row['time_received'] - target_t) * SPEED_OF_SOUND
        errors.append(error)
    errors_abs = np.abs(errors)

    return errors, np.mean(errors_abs), np.std(errors_abs)

if __name__ == '__main__':

    # 调用函数进行计算
    calculated_position = calculate_position(DATA_DF)
    print(f'监测设备信息:\n{DATA_DF}\n预测目标点信息:\n{calculated_position}')
    print(f'误差计算结果:{calculate_errors(DATA_DF, calculated_position)}')
    print(errors)
 
