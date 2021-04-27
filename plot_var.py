#读取数据
import pandas as pd
df = pd.read_csv(r'DATA黄鹤楼系列_2016-2021年_销量数据.csv',engine='python')
#print(df)

#1）导入模块
# 模型相关包
import statsmodels.api as sm
import statsmodels.stats.diagnostic
# 画图包
import matplotlib.pyplot as plt
# 其他包
import pandas as pd
import numpy as np

# 处理中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
# 处理符号显示不正常问题
plt.rcParams['axes.unicode_minus'] = False

#2）画序列相关图
fig = plt.figure(figsize=(12,8))
wuhan_cig =\
    df[(df['city_name']=='武汉市')&(df['bar_name']=='黄鹤楼(软蓝)')]['sale_num']\
        .tolist()
xiangyang_cig =\
    df[(df['city_name']=='襄阳市')&(df['bar_name']=='黄鹤楼(软蓝)')]['sale_num']\
        .tolist()
print(wuhan_cig)
print(xiangyang_cig)

plt.plot(wuhan_cig,'r',label='武汉')
plt.plot(xiangyang_cig,'g',label='襄阳')

# 两个行向量拼接到一起，形成一个两行的矩阵
x_y1 = np.r_[wuhan_cig, xiangyang_cig]
correlation = np.corrcoef(x_y1)
print('correlation ',correlation)
plt.title('Correlation: ' +str(correlation))

plt.grid(True)
plt.axis('tight')
plt.legend(loc=0)
plt.ylabel('Price')
plt.show()

#3）ADF单位根
def diff(xt,shift=1):
    xt_df = pd.DataFrame(xt)
    df_diff1_1 = xt_df.diff(shift)  ### 1阶差分，步长为1
    ret = df_diff1_1[shift:][0].tolist()
    print(ret)
    return ret

adfResult = sm.tsa.stattools.adfuller(diff(wuhan_cig))

output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used",
                                         "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"],
                                  columns=['value'])
output['value']['Test Statistic Value'] = adfResult[0]
output['value']['p-value'] = adfResult[1]
output['value']['Lags Used'] = adfResult[2]
output['value']['Number of Observations Used'] = adfResult[3]
output['value']['Critical Value(1%)'] = adfResult[4]['1%']
output['value']['Critical Value(5%)'] = adfResult[4]['5%']
output['value']['Critical Value(10%)'] = adfResult[4]['10%']

print(output)

#4）协整检验
#dummy_df = pd.get_dummies(df['烟品牌'],drop_first = False)
zhonghua_cig_made =\
    df[(df['city_name']=='武汉市')&(df['bar_name']=='黄鹤楼(软蓝)')]['sale_num']\
        .tolist()
result = sm.tsa.stattools.coint(zhonghua_cig_made,wuhan_cig)
print(result)

#https://blog.csdn.net/mooncrystal123/article/details/86736397

#建立对象，dataframe就是前面的data，varLagNum就是你自己定的滞后阶数
varLagNum = 1
df.index = df.month_id

#orgMod = sm.tsa.VARMAX(df[['brand_code','sale_num']],order=(varLagNum,0),trend='nc',exog=None)
orgMod = sm.tsa.AR(df['sale_num'])
#估计：就是模型
fitMod = orgMod.fit(maxlag= 20, ic= 'aic')
print('Lag: %s' % fitMod.k_ar)
print('Coefficients: %s' % fitMod.params)