import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib import rcParams
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

file_path = 'pb3\六年水沙月通量.csv'
monthly_flux = pd.read_csv(file_path, encoding='utf-8-sig')
monthly_flux['DATE'] = pd.to_datetime(monthly_flux[['Year', 'Month']].assign(DAY=1))
monthly_flux = monthly_flux.set_index('DATE')
monthly_flux = monthly_flux.drop(columns=['Year', 'Month'])
monthly_flux = monthly_flux[['水月通量', '沙月通量']]['2018-01':'2021-12']

water_flux = monthly_flux['水月通量']
sand_flux = monthly_flux['沙月通量']

water_flux_diff1 = water_flux.diff().dropna()
sand_flux_diff1 = sand_flux.diff().dropna()
water_flux_diff2 = water_flux_diff1.diff().dropna()
sand_flux_diff2 = sand_flux_diff1.diff().dropna()

plt.figure(figsize=(14, 6))
ax1 = plt.subplot(4, 2, 1)
water_flux.plot(ax=ax1, label='水月通量原值')
plt.legend()
ax3 = plt.subplot(4, 2, 3)
water_flux_diff2.plot(ax=ax3, label='水月通量二阶差分')
plt.legend()
ax5 = plt.subplot(4, 2, 5)
plot_acf(water_flux_diff2, ax=ax5, lags=len(water_flux_diff2)/2 - 1)
ax7 = plt.subplot(4, 2, 7)
plot_pacf(water_flux_diff2, ax=ax7, lags=len(water_flux_diff2)/2 - 1)

ax2 = plt.subplot(4, 2, 2)
sand_flux.plot(ax=ax2, label='沙月通量原值')
plt.legend()
ax4 = plt.subplot(4, 2, 4)
sand_flux_diff1.plot(ax=ax4, label='沙月通量一阶差分')
plt.legend()
ax6 = plt.subplot(4, 2, 6)
plot_acf(sand_flux_diff2, ax=ax6, lags=len(sand_flux_diff2)/2 - 1)
ax8 = plt.subplot(4, 2, 8)
plot_pacf(sand_flux_diff2, ax=ax8, lags=len(sand_flux_diff2)/2 - 1)


# 拟合 SARIMA 模型
def fit_sarima(series, order, seasonal_order):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    result = model.fit(disp=False)
    return result

# 水月通量的 SARIMA 模型
p, d, q = 4, 2, 1
P, D, Q, S = 0, 1, 1, 12
water_order = (p, d, q)
water_seasonal_order = (P, D, Q, S)
water_model = fit_sarima(water_flux, water_order, water_seasonal_order)

water_predict = water_model.get_prediction(start='2018-01', end='2023-12')
water_predict_mean = water_predict.predicted_mean
water_conf_int = water_predict.conf_int()
water_r2_score = r2_score(water_flux, water_predict_mean[:48])
print(f'{"="*10}水月通量预测{"="*10}')
print(water_predict_mean)
print(f'水月通量的 R² 值: {water_r2_score}')

plt.figure(figsize=(14, 6))
plt.plot(water_flux, label='水月通量原值')
plt.plot(water_predict_mean, label='水月通量预测值', linestyle='--')
plt.fill_between(water_predict_mean.index,
                 water_conf_int.iloc[:, 0],
                 water_conf_int.iloc[:, 1], color='k', alpha=0.1)
plt.legend()
plt.title('水月通量预测')

# 沙月通量的 SARIMA 模型
p, d, q = 1, 2, 2
P, D, Q, S = 1, 0, 0, 12
sand_order = (p, d, q)
sand_seasonal_order = (P, D, Q, S)
sand_model = fit_sarima(sand_flux, sand_order, sand_seasonal_order)

sand_predict = sand_model.get_prediction(start='2018-01', end='2023-12')
sand_predict_mean = sand_predict.predicted_mean
sand_conf_int = sand_predict.conf_int()
sand_r2_score = r2_score(sand_flux, sand_predict_mean[:48])
print(f'{"="*10}沙月通量预测{"="*10}')
print(sand_predict_mean)
print(f'沙月通量的 R² 值: {sand_r2_score}')

plt.figure(figsize=(14, 6))
plt.plot(sand_flux, label='沙月通量原值')
plt.plot(sand_predict_mean, label='沙月通量预测值', linestyle='--')
plt.fill_between(sand_predict_mean.index,
                 sand_conf_int.iloc[:, 0],
                 sand_conf_int.iloc[:, 1], color='k', alpha=0.1)
plt.legend()
plt.title('沙月通量预测')

predicted_result = pd.DataFrame({'水月通量': water_predict_mean[48:], '沙月通量': sand_predict_mean[48:]}, 
                                index=water_predict_mean[48:].index)
predicted_result.to_csv(r'pb3\2022-2023水沙月通量预测.csv', encoding='utf-8-sig')







plt.tight_layout()
plt.show()


















































# def rs_analysis(series):
#     N = len(series)
#     mean = np.mean(series)
#     Z = series - mean
#     Y = np.cumsum(Z)
#     R = np.max(Y) - np.min(Y)
#     S = np.std(series)
#     return R / S

# def hurst_exponent_rs(series):
#     N = len(series)
#     max_k = int(np.log2(N))
#     R_S = []
#     for k in range(2, max_k + 1):
#         n = 2 ** k
#         segments = [series[i:i + n] for i in range(0, N, n) if len(series[i:i + n]) == n]
#         R_S_values = [rs_analysis(segment) for segment in segments]
#         R_S.append(np.mean(R_S_values))
#     log_R_S = np.log(R_S)
#     log_n = np.log([2 ** k for k in range(2, max_k + 1)])
#     H, _ = np.polyfit(log_n, log_R_S, 1)
#     return H

# # R/S 分析
# water_rs = rs_analysis(monthly_flux['水月通量'])
# print(f'水月通量的 R/S 值: {water_rs}')
# sand_rs = rs_analysis(monthly_flux['沙月通量'])
# print(f'沙月通量的 R/S 值: {sand_rs}')

# # Hurst 指数
# water_hurst = hurst_exponent_rs(monthly_flux['水月通量'])
# print(f'水月通量的 Hurst 指数: {water_hurst}')
# sand_hurst = hurst_exponent_rs(monthly_flux['沙月通量'])
# print(f'沙月通量的 Hurst 指数: {sand_hurst}')





# # 设置时间索引
# monthly_flux.index = pd.date_range(start='2016-01', periods=len(monthly_flux), freq='M')

# # 拟合 SARIMA 模型
# def fit_sarima(series, order, seasonal_order):
#     model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
#     result = model.fit(disp=False)
#     return result

# # 水月通量的 SARIMA 模型
# water_order = (p, d, q)
# water_seasonal_order = (P, D, Q, S)
# water_model = fit_sarima(monthly_flux['水月通量'], water_order, water_seasonal_order)

# # 沙月通量的 SARIMA 模型
# sand_order = (p, d, q)
# sand_seasonal_order = (P, D, Q, S)
# sand_model = fit_sarima(monthly_flux['沙月通量'], sand_order, sand_seasonal_order)

# # 预测后两年（24个月）
# water_forecast = water_model.get_forecast(steps=24)
# sand_forecast = sand_model.get_forecast(steps=24)

# # 获取预测结果
# water_forecast_values = water_forecast.predicted_mean
# sand_forecast_values = sand_forecast.predicted_mean

# # 打印预测结果
# print("水月通量预测值:")
# print(water_forecast_values)
# print("沙月通量预测值:")
# print(sand_forecast_values)

# forecast_result = pd.DataFrame({'水月通量': water_forecast_values, '沙月通量': sand_forecast_values})
# forecast_result.to_csv(r'pb3\2022-2023水沙月通量预测.csv', encoding='utf-8-sig')
# print(forecast_result)

# # 可视化预测结果
# plt.figure(figsize=(12, 6))

# plt.subplot(2, 1, 1)
# plt.plot(monthly_flux['水月通量'], label='历史水月通量')
# plt.plot(water_forecast_values, label='预测水月通量', linestyle='--')
# plt.legend()
# plt.title('水月通量预测')

# plt.subplot(2, 1, 2)
# plt.plot(monthly_flux['沙月通量'], label='历史沙月通量')
# plt.plot(sand_forecast_values, label='预测沙月通量', linestyle='--')
# plt.legend()
# plt.title('沙月通量预测')

# plt.tight_layout()
# plt.show()





# # 设置时间索引
# monthly_flux.index = pd.date_range(start='2016-01', periods=len(monthly_flux), freq='M')

# # 分割数据集，使用前五年的数据进行训练
# train_data = monthly_flux['2018-01':'2020-12']
# test_data = monthly_flux['2021-01':]

# # 拟合 SARIMA 模型
# def fit_sarima(series, order, seasonal_order):
#     model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
#     result = model.fit(disp=False)
#     return result

# # 水月通量的 SARIMA 模型
# water_order = (p, d, q)
# water_seasonal_order = (P, D, Q, S)
# water_model = fit_sarima(train_data['水月通量'], water_order, water_seasonal_order)

# # 沙月通量的 SARIMA 模型
# sand_order = (p, d, q)
# sand_seasonal_order = (P, D, Q, S)
# sand_model = fit_sarima(train_data['沙月通量'], sand_order, sand_seasonal_order)

# # 预测2021年的数据
# water_forecast = water_model.get_forecast(steps=12)
# sand_forecast = sand_model.get_forecast(steps=12)

# # 获取预测结果
# water_forecast_values = water_forecast.predicted_mean
# sand_forecast_values = sand_forecast.predicted_mean

# # 评价预测结果
# water_rmse = np.sqrt(mean_squared_error(test_data['水月通量'], water_forecast_values))
# sand_rmse = np.sqrt(mean_squared_error(test_data['沙月通量'], sand_forecast_values))
# water_mape = np.mean(np.abs((test_data['水月通量'] - water_forecast_values) / test_data['水月通量'])) * 100
# sand_mape = np.mean(np.abs((test_data['沙月通量'] - sand_forecast_values) / test_data['沙月通量'])) * 100
# water_r2 = r2_score(test_data['水月通量'], water_forecast_values)
# sand_r2 = r2_score(test_data['沙月通量'], sand_forecast_values)

# # 打印评价结果
# print(f'水月通量预测的RMSE: {water_rmse}')
# print(f'水月通量预测的R²: {water_r2}')
# print(f'沙月通量预测的RMSE: {sand_rmse}')
# print(f'沙月通量预测的R²: {sand_r2}')

# # 可视化预测结果与实际数据对比
# plt.figure(figsize=(12, 6))

# plt.subplot(2, 1, 1)
# plt.plot(monthly_flux['水月通量'], label='历史水月通量')
# plt.plot(water_forecast_values, label='预测水月通量', linestyle='--', color='red')
# plt.plot(test_data['水月通量'], label='实际水月通量', linestyle='-', color='blue')
# plt.legend()
# plt.title('水月通量预测 vs 实际')

# plt.subplot(2, 1, 2)
# plt.plot(monthly_flux['沙月通量'], label='历史沙月通量')
# plt.plot(sand_forecast_values, label='预测沙月通量', linestyle='--', color='red')
# plt.plot(test_data['沙月通量'], label='实际沙月通量', linestyle='-', color='blue')
# plt.legend()
# plt.title('沙月通量预测 vs 实际')

# plt.tight_layout()
# plt.show()












# # 设置时间索引
# monthly_flux.index = pd.date_range(start='2016-01', periods=len(monthly_flux), freq='M')

# data = monthly_flux['2018-01':]
# train_data = data[:'2020-12']
# test_data = data['2021-01':]

# # 选择水月通量和沙月通量
# water_flux = data['水月通量']
# sand_flux = data['沙月通量']

# # 计算一阶差分
# water_flux_diff1 = water_flux.diff()
# sand_flux_diff1 = sand_flux.diff()

# # 绘制时间序列图
# plt.figure(figsize=(14, 14))

# 绘制水月通量
# plt.subplot(4, 2, 1)
# plt.plot(water_flux, label='水月通量原值')
# plt.legend()

# plt.subplot(4, 2, 3)
# plt.plot(water_flux_diff1, label='水月通量一阶差分', color='orange')
# plt.legend()

# # 绘制沙月通量
# plt.subplot(4, 2, 2)
# plt.plot(sand_flux, label='沙月通量原值')
# plt.legend()

# plt.subplot(4, 2, 4)
# plt.plot(sand_flux_diff1, label='沙月通量一阶差分', color='orange')
# plt.legend()

# # 绘制水月通量一阶差分的ACF和PACF图
# plt.subplot(4, 2, 5)
# plot_acf(water_flux_diff1, ax=plt.gca(), lags=2)
# plt.title('水月通量一阶差分的ACF')

# plt.subplot(4, 2, 7)
# plot_pacf(water_flux_diff1, ax=plt.gca(), lags=2)
# plt.title('水月通量一阶差分的PACF')

# # 绘制沙月通量一阶差分的ACF和PACF图
# plt.subplot(4, 2, 6)
# plot_acf(sand_flux_diff1, ax=plt.gca(), lags=2)
# plt.title('沙月通量一阶差分的ACF')

# plt.subplot(4, 2, 8)
# plot_pacf(sand_flux_diff1, ax=plt.gca(), lags=2)
# plt.title('沙月通量一阶差分的PACF')

# plt.tight_layout()
# plt.show()


