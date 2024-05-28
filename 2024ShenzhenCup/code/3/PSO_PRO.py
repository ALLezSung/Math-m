import pyswarms as ps
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


#导入初始数据
data = {
    'longitude': [110.241, 110.780, 110.712, 110.251, 110.524, 110.467, 110.047],
    'latitude': [27.204, 27.456, 27.785, 27.825, 27.617, 27.921, 27.121],
    'altitude': [824, 727, 742, 850, 786, 678, 575],
    'time': [[100.767, 164.229, 214.850, 270.065],
             [92.453, 112.220, 169.362, 196.583],
             [75.560, 110.696, 156.936, 188.020],
             [94.653, 141.409, 196.517, 258.985],
             [78.600, 86.216, 118.443, 126.669],
             [67.274, 166.270, 175.482, 266.871],
             [103.738, 163.024, 206.789, 210.306]]
}

DATA_DF = pd.DataFrame(data, index=['A', 'B', 'C', 'D', 'E', 'F', 'G'])

#数据预处理
DATA_DF['x'] = DATA_DF['longitude'] * X_LENGTH
DATA_DF['y'] = DATA_DF['latitude'] * Y_LENGTH
DATA_DF['z'] = DATA_DF['altitude'] / 1000

REQUIREMENTS = 4
REQUIREMENTS -= 1

lizishu = 30
RESULT_POINTS = pd.DataFrame(columns=['x', 'y', 'z', 'time_explosion', 'sum_error'])
errors = np.zeros(lizishu)

for Time_Num in range(len(DATA_DF['time'].iloc[0])):

    POINT = {
        'x': 0,
        'y': 0,
        'z': 0,
        'time_explosion': 0,
        'sum_error': 42
    }

    
    # 定义目标函数
    def objective_function(particles):
        tol = 1e-2
        n_particles = particles.shape[0]  # 获取粒子的数量

        for NUMBER, _particle in enumerate(particles):
            _particle = _particle.astype(int)

            points = DATA_DF[['x', 'y', 'z']].to_numpy()[:REQUIREMENTS,:]

            times = [DATA_DF['time'].iloc[0][Time_Num]]
            for _ in range(len(DATA_DF) - 8 + REQUIREMENTS):
                times.append(DATA_DF['time'].iloc[_ + 1][_particle[_]])
            times_received = np.array(times)

            def objective(variables, points, times_received):
                signal_origin = variables[:3]
                signal_time = variables[3]
                euclid_distances = np.linalg.norm(points - signal_origin, axis=1)
                sound_distances = (times_received - signal_time) * SPEED_OF_SOUND
                
                error = np.sum(np.abs(euclid_distances - sound_distances))
                if error < POINT['sum_error']:
                    POINT['x'] = signal_origin[0]
                    POINT['y'] = signal_origin[1]
                    POINT['z'] = signal_origin[2]
                    POINT['time_explosion'] = signal_time
                    POINT['sum_error'] = error

                errors[NUMBER] = error
                return error
            
            bounds = [(-180*97.304, 180*97.304), (-90*111.263, 90*111.263), (0, 10000), (0, np.min(times_received))]

            # 初始猜测
            x0 = np.zeros(4)
            x0[:3] = np.mean(points, axis=0)  # 假设初始位置为所有监测设备位置的平均值
            x0[3] = 0  # 假设初始信号发出时间为0

            # 实施优化
            result = minimize(objective, x0, 
                            args=(points, times_received), 
                            method='L-BFGS-B', options={'gtol': tol, 'maxiter': 100},
                            bounds=bounds
                            )
            if result.success:
                pass
            else:
                pass
        return errors
        
    # 设置优化器选项
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    bounds = (np.zeros(REQUIREMENTS), np.full(REQUIREMENTS, 4))

    # 创建局部最佳优化器
    optimizer = ps.single.GlobalBestPSO(
        n_particles=lizishu,          # 粒子的数量
        dimensions=REQUIREMENTS,           
        options=options,         
        bounds=bounds            
    )

    # 执行优化
    cost, pos = optimizer.optimize(objective_function, iters=5)

    print(f"第{Time_Num + 1}个残骸的时间组合为：{[Time_Num] + pos.astype(int).tolist()}, 平均误差：{cost/REQUIREMENTS}")
    
    POINT = pd.DataFrame(POINT,index=[Time_Num])
    RESULT_POINTS = pd.concat([RESULT_POINTS, POINT])

print(RESULT_POINTS)


# answers = [[0, 1, 2, 1, 0, 2, 1], [1, 3, 3, 1, 2, 2, 2], [2, 3, 3, 1, 2, 2, 3], [3, 3, 3, 1, 2, 2, 3]]

