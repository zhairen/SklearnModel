#读取数据
import pandas as pd
df = pd.read_excel(r'D:\风险工作\招商银行经营贷\烟预测\销量.xlsx')
print(df)

#1）导入模块
# 模型相关包
import statsmodels.api as sm
import statsmodels.stats.diagnostic
# 画图包
import matplotlib.pyplot as plt
# 其他包
import pandas as pd
import numpy as np

#2）画序列相关图
fig = plt.figure(figsize=(12,8))
zhonghua_cig = df[df['烟品牌']=='中华']['销售量'].tolist()
yuxi_cig = df[df['烟品牌']=='玉溪']['销售量'].tolist()

plt.plot(zhonghua_cig,'r',label='中华')
plt.plot(yuxi_cig,'g',label='玉溪')

# 两个行向量拼接到一起，形成一个两行的矩阵
x_y1 = np.r_[zhonghua_cig, yuxi_cig]
correlation = np.corrcoef(x_y1)
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

adfResult = sm.tsa.stattools.adfuller(diff(zhonghua_cig))

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
zhonghua_cig_made = df[df['烟品牌']=='中华']['投放量'].tolist()

result = sm.tsa.stattools.coint(zhonghua_cig_made,zhonghua_cig)
print(result)

#https://blog.csdn.net/mooncrystal123/article/details/86736397

#建立对象，dataframe就是前面的data，varLagNum就是你自己定的滞后阶数
varLagNum = 1

orgMod = sm.tsa.VARMAX(df,order=(varLagNum,0),trend='nc',exog=None)
#估计：就是模型
fitMod = orgMod.fit(maxiter=1000,disp=False)
# 打印统计结果
print(fitMod.summary())
# 获得模型残差
resid = fitMod.resid
result = {'fitMod':fitMod,'resid':resid}
