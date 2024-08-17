import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# 读取数据
excel_file = 'pb1/水沙数据.xlsx'
dfs = pd.read_excel(excel_file, sheet_name=None)


# 简化表头，填充日期
new_headers = ['年', '月', '日', '时间', '水位', '流量', '含沙量'] 
for sheet_name, df in dfs.items():
    if len(df.columns) == len(new_headers):
        df.columns = new_headers
    else:
        print(f"Sheet {sheet_name} 的列数与新表头的列数不匹配，无法替换表头。")
    df[['年', '月', '日']] = df[['年', '月', '日']].ffill().astype(int)


# 获取含沙量数据
def get_sand_content(dfs):
    progressed_dfs = []
    for df in dfs.values():
        grouped = df.groupby(['年', '月', '日'])
        merged_data = []
        for name, group in grouped:
            row_8am = group[group['时间'] == '8:00']
            if not row_8am.empty:
                sand_content = row_8am['含沙量'].values[0]
                merged_data.append({'年': name[0], '月': name[1], '日': name[2], '含沙量': sand_content})
            else:
                merged_data.append({'年': name[0], '月': name[1], '日': name[2], '含沙量': None})
        merged_df = pd.DataFrame(merged_data)
        progressed_dfs.append(merged_df)
    return pd.concat(progressed_dfs)
sand = get_sand_content(dfs)
sand['含沙量'] = sand['含沙量'].interpolate(method='linear')


# # 绘图
# sand.columns = ['year', 'month', 'day', '含沙量']
# sand['日期'] = pd.to_datetime(sand[['year', 'month', 'day']])
# rcParams['font.sans-serif'] = ['SimHei']
# rcParams['axes.unicode_minus'] = False
# plt.figure(figsize=(10, 6))
# plt.plot(sand['日期'], sand['含沙量'], marker='o', linestyle='None', markersize=1)
# plt.xlabel('日期')
# plt.ylabel('含沙量')
# plt.title('含沙量随时间的变化')
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# 提取同时拥有含沙量数据和流量数据的行
def get_sand_and_flow_content(dfs):
    merged_data = []
    for df in dfs.values():
        for _, row in df.iterrows():
            sand_content = row['含沙量']
            flow_content = row['流量']
            if pd.notna(sand_content) and pd.notna(flow_content):
                merged_data.append({
                    '年': row['年'],
                    '月': row['月'],
                    '日': row['日'],
                    '时间': row['时间'],
                    '含沙量': sand_content,
                    '流量': flow_content
                })
    return pd.DataFrame(merged_data)
sand_and_flow = get_sand_and_flow_content(dfs)
correlation = sand_and_flow['含沙量'].corr(sand_and_flow['流量'])
print(f"含沙量与流量的皮尔逊相关系数: {correlation}")

small_sand_and_flow = sand_and_flow[sand_and_flow['流量'] < 2000]
correlation = small_sand_and_flow['含沙量'].corr(small_sand_and_flow['流量'])
print(f"含沙量与流量的皮尔逊相关系数(流量小于2000): {correlation}")

coefficients = np.polyfit(small_sand_and_flow['流量'], small_sand_and_flow['含沙量'], 1)
poly = np.poly1d(coefficients)
print(f"拟合直线方程: y = {coefficients[0]}x + {coefficients[1]}")


# # 绘图
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 6))
plt.scatter(sand_and_flow['流量'], sand_and_flow['含沙量'], c='blue', s=2, label='数据点')
plt.plot(sand_and_flow['流量'], poly(sand_and_flow['流量']), color='red', 
         label=f'拟合直线: y = {coefficients[0]:.4f}x + {coefficients[1]:.4f}')
plt.xlabel('流量')
plt.ylabel('含沙量')
plt.title('含沙量与流量相关性')
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()


# 提取同时拥有含沙量数据和水位数据的行
def get_sand_and_depth_content(dfs):
    merged_data = []
    for df in dfs.values():
        for _, row in df.iterrows():
            sand_content = row['含沙量']
            depth_content = row['水位']
            if pd.notna(sand_content) and pd.notna(depth_content):
                merged_data.append({
                    '年': row['年'],
                    '月': row['月'],
                    '日': row['日'],
                    '时间': row['时间'],
                    '含沙量': sand_content,
                    '水位': depth_content
                })
    return pd.DataFrame(merged_data)
sand_and_depth = get_sand_and_depth_content(dfs)
correlation = sand_and_depth['含沙量'].corr(sand_and_depth['水位'])
print(f"含沙量与水位的皮尔逊相关系数: {correlation}")

small_sand_and_depth = sand_and_depth[(sand_and_depth['水位'] > 42) & (sand_and_depth['含沙量'] < 20)]
correlation = small_sand_and_depth['含沙量'].corr(small_sand_and_depth['水位'])
print(f"含沙量与水位的皮尔逊相关系数(含沙量小于20): {correlation}")

coefficients = np.polyfit(small_sand_and_depth['水位'], small_sand_and_depth['含沙量'], 1)
poly = np.poly1d(coefficients)
print(f"拟合直线方程: y = {coefficients[0]}x + {coefficients[1]}")

# # 绘图
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 6))
plt.scatter(sand_and_depth['水位'], sand_and_depth['含沙量'], c='blue', s=2, label='数据点')
plt.plot(sand_and_depth['水位'], poly(sand_and_depth['水位']), color='red', 
         label=f'拟合直线: y = {coefficients[0]:.4f}x + {coefficients[1]:.4f}')
plt.xlabel('水位')
plt.ylabel('含沙量')
plt.title('含沙量与水位相关性')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


def fit_log_log_relationship(x, y, plot=False):
    """
    使用最小二乘法拟合 ln(y) = a * ln(x) + b，并返回拟合系数 a 和 b。
    
    参数:
    x -- 自变量数据（例如水位或流量）
    y -- 因变量数据（例如含沙量）
    plot -- 是否绘制拟合结果，默认为 False
    
    返回:
    a, b -- 拟合系数
    """
    # 取对数
    log_x = np.log(x)
    log_y = np.log(y)
    
    # 使用 numpy.polyfit 进行线性拟合
    coefficients = np.polyfit(log_x, log_y, 1)
    a, b = coefficients    

    from scipy.stats import linregress
    
    # 假设 log_x 和 log_y 已经计算好
    log_y = np.log(y)
    
    # 使用 numpy.polyfit 进行线性拟合
    coefficients = np.polyfit(log_x, log_y, 1)
    a, b = coefficients
    
    # 使用 scipy.stats.linregress 计算拟合优度和 p-value
    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
    
    # 打印拟合优度和 p-value
    print(f"拟合优度 (R²): {r_value**2}")
    print(f"T检验的相伴概率 (p-value): {p_value}")
    
    if plot:
        # 绘图
        rcParams['font.sans-serif'] = ['SimHei']
        rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 6))
        plt.scatter(log_x, log_y, c='blue', s=2, label='数据点')
        plt.plot(log_x, a * log_x + b, color='red', label='拟合线')
        plt.legend()
        plt.show()
    
    if plot:
        # 绘图
        rcParams['font.sans-serif'] = ['SimHei']
        rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 6))
        plt.scatter(log_x, log_y, c='blue', s=2, label='数据点')
        plt.plot(log_x, a * log_x + b, color='red', label=f'拟合直线: ln(y) = {a:.4f} * ln(x) + {b:.4f}')
        plt.xlabel('ln(自变量)')
        plt.ylabel('ln(因变量)')
        plt.title('ln(因变量) 与 ln(自变量) 的拟合')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return a, b

a, b = fit_log_log_relationship(sand_and_flow['流量'], sand_and_flow['含沙量'])
print(f"拟合结果: ln(含沙量) = {a} * ln(流量) + {b}")
a, b = fit_log_log_relationship(sand_and_depth['水位'], sand_and_depth['含沙量'])
print(f"拟合结果: ln(含沙量) = {a} * ln(水位) + {b}")

sand.to_excel('pb1/含沙量.xlsx', index=False)
sand_and_flow.to_excel('pb1/含沙量与流量.xlsx', index=False)
sand_and_depth.to_excel('pb1/含沙量与水位.xlsx', index=False)