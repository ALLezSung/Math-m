from matplotlib import rcParams
import pandas as pd
import matplotlib.pyplot as plt
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

file_path = 'pb4\部分监测点数据.csv'

# 读取 CSV 文件
data = pd.read_csv(file_path)
data.columns = ['Date', 'Distance', 'Level', 'Depth', 'Measurement_depth', 'Speed', 'Sand']

# # 1. 绘制起始距离与流速的散点图
# scatter1 = data[['Distance', 'Speed']].copy()
# scatter1['Distance'] = scatter1['Distance'].ffill()
# scatter1.dropna(subset=['Speed'], inplace=True)
# print(scatter1)
# plt.scatter(scatter1['Distance'], scatter1['Speed'], s=4)
# plt.xlabel('Distance')
# plt.ylabel('Speed')
# plt.title('Scatter plot of starting distance and flow velocity')

# # 2. 绘制测量深度与流速的散点图
# scatter2 = data[['Measurement_depth', 'Speed']].copy()
# scatter2.dropna(subset=['Speed'], inplace=True)
# scatter2.dropna(subset=['Measurement_depth'], inplace=True)
# print(scatter2)
# plt.figure()
# plt.scatter(scatter2['Measurement_depth'], scatter2['Speed'], s=4)
# plt.xlabel('Measurement depth')
# plt.ylabel('Speed')
# plt.title('Scatter plot of measurement depth and flow velocity')

# # 3. 绘制起点距离、测量深度和流速的三维散点图
# scatter3 = data[['Distance', 'Measurement_depth', 'Speed', 'Depth', 'Date']].copy()
# scatter3['Date'] = scatter3['Date'].ffill()
# scatter3['Distance'] = scatter3['Distance'].ffill()
# scatter3['Depth'] = scatter3['Depth'].ffill()
# scatter3.dropna(subset=['Speed', 'Measurement_depth', 'Date'], inplace=True)
# scatter3['Height'] = scatter3['Depth'] - scatter3['Measurement_depth']

# scatter3 = scatter3[(scatter3['Measurement_depth'] < 20) & (scatter3['Speed'] < 15) 
#                     & (scatter3['Distance'] < 20000) & (scatter3['Distance'] > 0)]
# scatter3 = scatter3[scatter3['Date'] < '2023-01-01']

# # 将日期转换为数值以便着色
# scatter3['Date'] = pd.to_datetime(scatter3['Date'])
# scatter3['Date_numeric'] = scatter3['Date'].map(pd.Timestamp.toordinal)

# plt.figure()
# ax = plt.axes(projection='3d')
# sc = ax.scatter3D(scatter3['Distance'], scatter3['Height'], scatter3['Speed'], c=scatter3['Date_numeric']-736500, cmap='viridis', s=4)
# ax.set_xlabel('Distance')
# ax.set_ylabel('Height')
# ax.set_zlabel('Speed')
# ax.set_title('3D scatter plot of starting distance, measurement depth and flow velocity')

# # 添加颜色条
# cbar = plt.colorbar(sc, ax=ax, pad=0.1)
# cbar.set_label('Date')

# plt.figure()
# ax = plt.subplot()
# ax.plot(scatter3['Height'], scatter3['Speed'], 'o', markersize=2)
# ax.set_xlabel('Height')
# ax.set_ylabel('Speed')
# ax.set_title('Scatter plot of height and flow velocity')

# plt.figure()
# ax = plt.subplot()
# ax.plot(scatter3['Distance'], scatter3['Speed'], 'o', markersize=2)
# ax.set_xlabel('Distance')
# ax.set_ylabel('Speed')
# ax.set_title('Scatter plot of distance and flow velocity')

# plt.tight_layout()
# plt.show()


scatter4 = data[['Date', 'Distance', 'Level', 'Depth']].copy()
scatter4 = scatter4.dropna(subset=['Distance', 'Depth'])
scatter4['Level'] = scatter4['Level'].ffill()
scatter4['Date'] = scatter4['Date'].ffill()
scatter4 = scatter4[scatter4['Depth'] != 0]
scatter4['Date'] = pd.to_datetime(scatter4['Date'])
scatter4['altitude'] = scatter4['Level'] - scatter4['Depth']

scatter4.to_csv('pb4\水站水底高程表.csv', index=False, encoding='utf-8-sig')

average_altitude = scatter4.groupby(scatter4['Date'].dt.date)['altitude'].mean()

plt.figure(figsize=(10, 6))
average_altitude.plot(kind='line')
plt.title('Average Altitude over Time')
plt.xlabel('Date')
plt.ylabel('Average Altitude')
plt.grid(True)
plt.tight_layout()
plt.show()
