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


Num_p = 3000
def PSO_function(df):

    points = df[['x', 'y', 'z']].to_numpy()
    times_received = df['time_received'].to_numpy()
    errors = np.zeros(Num_p)

    def objective_function(particles):
        signal_origin = particles[:, :3]
        signal_time = particles[:, 3]
        for _, point in enumerate(particles):
            euclid_distances = np.linalg.norm(points - signal_origin[_], axis=1)
            sound_distances = (times_received - signal_time[_]) * SPEED_OF_SOUND
            errors[_] = np.mean(np.abs(euclid_distances - sound_distances))
        return errors
    

    options = {'c1': 0.4, 'c2': 0.6, 'w':0.5}
    bounds = ([109*X_LENGTH ,26*Y_LENGTH ,0 ,0],[111*X_LENGTH,28*Y_LENGTH ,10 ,min(DATA_DF['time_received'])])


    optimizer = ps.single.GlobalBestPSO(
        n_particles=Num_p,          # 粒子的数量
        dimensions=4,           
        options=options,         
        bounds=bounds            
    )

    # 执行优化
    cost, pos = optimizer.optimize(objective_function, iters=100)

    RESULT_POINT = pd.DataFrame({'longitude': pos[0]/97.304, 'latitude': pos[1]/111.263, 'altitude': pos[2]*1000, 
                                    'time_explosion': pos[3], 'x':pos[0], 'y':pos[1], 'z':pos[2]}, index=['Position'])
    
    print(RESULT_POINT, "\n误差(km):", cost)    

    return RESULT_POINT

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
      calculated_position = PSO_function(DATA_DF)   
      plot_devices_and_target(DATA_DF, calculated_position)

