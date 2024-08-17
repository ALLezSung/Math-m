from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.dates as mdates
from scipy import stats
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# # 读取Excel文件
# file_path = 'pb2\含沙量与流量.xlsx'
# df = pd.read_excel(file_path)

# df.columns = ['Year', 'Month', 'Day', 'hour', 'Sand', 'Flow']
# df['日期'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# # 提取月份和年份
# df['年份'] = df['日期'].dt.year
# df['月份'] = df['日期'].dt.month
# df['日期'] = df['日期'].dt.date

# # 筛选
# latest_year = df['年份'].max()
# df_recent_six_years = df[df['年份'] >= latest_year - 5]

# def calculate_daily_flux(group):
#     daily_flow_mean = group['Flow'].mean()
#     daily_water_flux = daily_flow_mean * 24 * 3600  # 每天的水日通量
#     daily_sand_mean = group['Sand'].mean()
#     daily_sand_flux = daily_sand_mean * daily_water_flux  # 每天的沙日通量
#     return pd.Series({'水日通量': daily_water_flux, '沙日通量': daily_sand_flux})
# daily_flux = df_recent_six_years.groupby(['Year', 'Month', 'Day']).apply(calculate_daily_flux)
# monthly_flux = daily_flux.groupby(['Year', 'Month']).sum()


# monthly_flux.columns = ['月水通量', '月沙通量']
# monthly_flux.to_excel('pb2\月水通量和沙通量.xlsx')

# print(monthly_flux)










# file_path = 'pb2\月水通量和沙通量.xlsx'

# df = pd.read_excel(file_path)
# df.columns = ['Year', 'Month', 'WaterFlux', 'SandFlux']

# df['WaterFlux'] = df['WaterFlux'] / 100000000  # 将水通量转换为 0.1Bm^3
# df['SandFlux'] = df['SandFlux'] / 1000000000  # 将沙通量转换为 Mt

# df['Datetime'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1))
# df['年份'] = df['Datetime'].dt.year
# df['月份'] = df['Datetime'].dt.month

# print(df[['Datetime', 'WaterFlux']].dropna().head())

# plt.figure(figsize=(10, 6))
# plt.plot(df['Datetime'], df['WaterFlux'], label='水通量(亿立方米)')
# plt.plot(df['Datetime'], df['SandFlux'], label='沙通量(百万吨)')
# plt.xlabel('时间')
# plt.ylabel('通量')
# plt.legend()
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.gcf().autofmt_xdate()
# plt.xticks(fontsize=5)
# # plt.savefig('pb2\水沙通量时间序列图.svg')

# # 按月份分组并计算统计量
# monthly_stats = df.groupby(['月份']).agg({
# 	'WaterFlux': ['mean', 'var', 'max', 'min', lambda x: x.max() - x.min()],
# 	'SandFlux': ['mean', 'var', 'max', 'min', lambda x: x.max() - x.min()]
# })
# # 重命名列
# monthly_stats.columns = [
#     '水均值', '水方差', '水最大值', '水最小值', '水极差',
#     '沙均值', '沙方差', '沙最大值', '沙最小值', '沙极差'
# ]
# print(monthly_stats)
# monthly_stats.to_excel('pb2\月水沙通量统计.xlsx')

# plt.show()









# file_path = 'pb2\含沙量与流量.xlsx'
# df = pd.read_excel(file_path)

# df.columns = ['Year', 'Month', 'Day', 'hour', 'Sand', 'Flow']
# df['日期'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# # 提取月份和年份
# df['年份'] = df['日期'].dt.year
# df['月份'] = df['日期'].dt.month
# df['日期'] = df['日期'].dt.date

# # 筛选
# latest_year = df['年份'].max()
# df_recent_six_years = df[df['年份'] >= latest_year - 5]

# def calculate_daily_flux(group):
#     year = group['Year'].iloc[0]
#     month = group['Month'].iloc[0]
#     day = group['Day'].iloc[0]
#     daily_flow_mean = group['Flow'].mean()
#     daily_water_flux = daily_flow_mean * 24 * 3600  # 每天的水日通量
#     daily_sand_mean = group['Sand'].mean()
#     daily_sand_flux = daily_sand_mean * daily_water_flux  # 每天的沙日通量
#     return pd.Series({'Year': year, 'Month': month, 'Day': day, '水日通量': daily_water_flux, '沙日通量': daily_sand_flux})
# daily_flux = df_recent_six_years.groupby(['Year', 'Month', 'Day']).apply(calculate_daily_flux).reset_index(drop=True)
# daily_flux['日期'] = pd.to_datetime(daily_flux[['Year', 'Month', 'Day']])


# full_date_range = pd.date_range(start=df_recent_six_years['日期'].min(), end=df_recent_six_years['日期'].max())
# daily_flux = daily_flux.set_index('日期').reindex(full_date_range)
# daily_flux = daily_flux.interpolate(method='linear')


# daily_flux = daily_flux.drop(columns=['Year', 'Month', 'Day'])
# daily_flux['Year'] = daily_flux.index.year
# daily_flux['Month'] = daily_flux.index.month
# daily_flux['Day'] = daily_flux.index.day

# desired_order = ['Year', 'Month', 'Day'] + [col for col in daily_flux.columns if col not in ['Year', 'Month', 'Day']]
# daily_flux = daily_flux[desired_order]
# daily_flux.to_csv('pb2\六年内每天的水沙日通量.csv', encoding='utf-8-sig')

# print(daily_flux)






# file_path = 'pb2\六年水沙日通量.csv'
# df = pd.read_csv(file_path, index_col=0)
# df.columns = ['Year', 'Month', 'Day', 'WaterFlux', 'SandFlux']

# df['WaterFlux'] = df['WaterFlux'] / 100000000  # 亿立方米
# df['SandFlux'] = df['SandFlux'] / 1000000000  # 百万吨

# df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
# df['年份'] = df['Datetime'].dt.year
# df['月份'] = df['Datetime'].dt.month

# plt.figure(figsize=(10, 6))
# plt.plot(df['Datetime'], df['WaterFlux'], label='水通量(亿立方米)')
# plt.plot(df['Datetime'], df['SandFlux'], label='沙通量(百万吨)')
# plt.xlabel('时间')
# plt.ylabel('通量')
# plt.title('水沙通量时间序列图')
# plt.legend()
# def format_date(x, pos=None):
#     date = mdates.num2date(x)
#     if date.month == 1:
#         return date.strftime('%Y-%m')
#     else:
#         return date.strftime('%m')
# plt.gca().xaxis.set_major_formatter(FuncFormatter(format_date))
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
# plt.gcf().autofmt_xdate()
# plt.xticks(fontsize=7)
# plt.savefig('pb2\水沙通量时间序列图.svg')

# plt.show()





def sk(data):
    n=len(data)
    Sk     = [0]
    UFk    = [0]
    s      =  0
    E      = [0]
    Var    = [0]
    for i in range(1,n):
        for j in range(i):
            if data[i] > data[j]:
                s = s+1
            else:
                s = s+0
        Sk.append(s)
        E.append((i+1)*(i+2)/4 )                     # Sk[i]的均值
        Var.append((i+1)*i*(2*(i+1)+5)/72 )            # Sk[i]的方差
        UFk.append((Sk[i]-E[i])/np.sqrt(Var[i]))
    UFk=np.array(UFk)
    return UFk
#a为置信度
def MK(data,a):
    ufk=sk(data)          #顺序列
    ubk1=sk(data[::-1])   #逆序列
    ubk=-ubk1[::-1]        #逆转逆序列
    #输出突变点的位置
    p=[]
    u=ufk-ubk
    for i in range(1,len(ufk)):
        if u[i-1]*u[i]<0:
            p.append(i)
    if p:
        print("突变点位置：",p)
    else:
        print("未检测到突变点")
    #画图
    conf_interval = stats.norm.interval(a, loc=0, scale=1)   #获取置信区间
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']#解决中文不显示问题
    plt.rcParams['axes.unicode_minus']=False # 解决负号不显示问题
    plt.figure(figsize=(10,6))
    plt.plot(range(len(data)),ufk,label = 'UFk',color = 'r')
    plt.plot(range(len(data)),ubk,label = 'UBk',color = 'b')
    plt.ylabel('UFk-UBk', fontsize=25)
    plt.xlabel('月数', fontsize=18)
    plt.title('Mann-Kendall检验（水通量）', fontsize=25)
    x_lim = plt.xlim()
    plt.ylim([-6, 7])
    plt.plot(x_lim, [conf_interval[0], conf_interval[0]], 'm--', label='95%显著区间')
    plt.plot(x_lim, [conf_interval[1], conf_interval[1]], 'm--')
    plt.axhline(0, ls="--", c="k")
    plt.legend(loc='upper center', frameon=False, ncol=3, fontsize=20)  # 图例
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()

file_path = 'pb2\六年水沙月通量.csv'
df = pd.read_csv(file_path, index_col=0)
df.columns = ['Month', 'WaterFlux', 'SandFlux']
water_flux = df['WaterFlux'].values
sand_flux = df['SandFlux'].values

for data in [water_flux]:
    MK(data,0.95)






