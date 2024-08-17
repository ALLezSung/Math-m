import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# # 读取CSV文件
# data = pd.read_csv('pb4\断面数据.csv')

# data1 = data[['2016-06-08 00:00:00', 'Unnamed: 1']].copy()
# data2 = data[['2016-10-20 00:00:00', 'Unnamed: 3']].copy()
# data3 = data[['2017-05-11 00:00:00', 'Unnamed: 5']].copy()
# data4 = data[['2017-09-05 00:00:00', 'Unnamed: 7']].copy()
# data5 = data[['2018-09-13 00:00:00', 'Unnamed: 9']].copy()
# data6 = data[['2019-04-13 00:00:00', 'Unnamed: 11']].copy()
# data7 = data[['2019-10-15 00:00:00', 'Unnamed: 13']].copy()
# data8 = data[['2020-03-19 00:00:00', 'Unnamed: 15']].copy()
# data9 = data[['2021-03-14 00:00:00', 'Unnamed: 17']].copy()

# data1.columns, data2.columns, data3.columns = ['Distance', 'Height'], ['Distance', 'Height'], ['Distance', 'Height']
# data4.columns, data5.columns, data6.columns = ['Distance', 'Height'], ['Distance', 'Height'], ['Distance', 'Height']
# data7.columns, data8.columns, data9.columns = ['Distance', 'Height'], ['Distance', 'Height'], ['Distance', 'Height']

# data1, data2, data3 = data1[1:].dropna(), data2[1:].dropna(), data3[1:].dropna()
# data4, data5, data6 = data4[1:].dropna(), data5[1:].dropna(), data6[1:].dropna()
# data7, data8, data9 = data7[1:].dropna(), data8[1:].dropna(), data9[1:].dropna()

# data_list = [data1, data2, data3, data4, data5, data6, data7, data8, data9]
# date_list = ['2016-06-08', '2016-10-20', '2017-05-11', '2017-09-05', '2018-09-13', '2019-04-13', '2019-10-15', '2020-03-19', '2021-03-14']


# # 创建三维图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 遍历每个时间点的数据
# for data, date in zip(data_list, date_list):
#     x = data['Distance'].astype(float)
#     y = [mdates.date2num(pd.to_datetime(date))] * len(data)  # 日期作为y轴
#     z = data['Height'].astype(float)
    
#     ax.plot(x, y, z, label=f'日期 {date}')

# ax.set_xlabel('距离')
# ax.set_xlim(min(data1['Distance'].astype(float)), 2000)
# ax.set_ylabel('日期')
# ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# ax.set_zlabel('高程')

# # ax.legend()
# plt.tight_layout()
# plt.show()



# import plotly.graph_objects as go
# fig = go.Figure()

# # 遍历每个时间点的数据
# for data, date in zip(data_list, date_list):
#     x = data['Distance'].astype(float)
#     y = [date] * len(data)  # 日期作为y轴，保持为字符串格式
#     z = data['Height'].astype(float)
    
#     fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name=f'日期 {date}'))

# fig.update_layout(
#     scene=dict(
#         xaxis=dict(
#             title='距离',
#             range=[-500, 7000]  
#         ),
#         yaxis=dict(
#             title='日期',
#             type='category'  
#         ),
#         zaxis=dict(
#             title='高程'
#         ),
#         aspectratio=dict(
#             x=2.5,  
#             y=1,
#             z=1
#         )
#     )
# )

# # 保存为HTML文件
# fig.write_html('水位图.html')

# fig.show()








# for i in range(len(data_list)):
#     data_list[i]['Distance'] = pd.to_numeric(data_list[i]['Distance'], errors='coerce')
#     data_list[i] = data_list[i][data_list[i]['Distance'] <= 4500]

# from scipy.interpolate import interp1d
# import numpy as np
# distance_min = 0
# distance_max = 4500
# distance_step = 10
# distances = np.arange(distance_min, distance_max + distance_step, distance_step)

# # 初始化一个空列表存储每个时间点的插值结果
# interpolated_elevations = []

# for df in data_list:
#     # 对 Distance 和 Elevation 列进行插值
#     f = interp1d(df['Distance'], df['Height'], bounds_error=False, fill_value="extrapolate")
#     interpolated_elevation = f(distances)
#     interpolated_elevations.append(interpolated_elevation)

# # 将插值结果转换为 NumPy 数组以便于计算
# interpolated_elevations = np.array(interpolated_elevations)

# # 创建 DataFrame，行索引为距离点，列索引为测量日期
# average_elevations_df = pd.DataFrame(interpolated_elevations.T, index=distances, columns=date_list)

# average_elevations_df.to_csv('pb4\断面数据插值表.csv')


from datetime import datetime
file_path = 'pb4\断面数据插值表.csv'
df = pd.read_csv(file_path, index_col=0)
df = df.dropna().T
df['时间'] = pd.to_datetime(df.index)

# 绘制时间和平均高程的折线图
plt.figure(figsize=(10, 6))
plt.plot(df['时间'], df['avg'], marker='o', label='平均高程')

# 拟合前五个平均值数据
first_five = df.head(5)
x = first_five['时间'].map(datetime.timestamp).values
y = first_five['avg'].values
coefficients = np.polyfit(x, y, 1)
linear_fit = np.poly1d(coefficients)

# 预测2026-06-08的平均高程
future_date = datetime(2026, 6, 8)
future_timestamp = datetime.timestamp(future_date)
predicted_value = linear_fit(future_timestamp)

# 扩展fit_x的范围到预测日期
fit_x = np.linspace(df['时间'].map(datetime.timestamp).min(), future_timestamp, num=1000)
fit_y = linear_fit(fit_x)

# 在图表中绘制拟合直线
extended_dates = pd.to_datetime(fit_x, unit='s')
plt.plot(extended_dates, fit_y, label='无调水调沙预测直线', linestyle='--')

# 打印预测结果
print(f"预测2026-06-08的平均高程: {predicted_value:.2f}")

# 设置图表标签和标题
plt.xlabel('时间')
plt.ylabel('平均高程')
plt.title('水文站平均高程时间序列图')
plt.legend()
plt.grid(True)

# 显示图表
plt.show()