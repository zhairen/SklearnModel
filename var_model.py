#-*- coding:utf-8 -*-
#读取数据
import pandas as pd
df_read = pd.read_csv(r'D:/DATA黄鹤楼系列_2016-2021年_销量数据.csv',engine='python')
#print(df)
df = df_read[df_read['sshengfen']=='湖北省']

df_store= pd.read_csv(r'D:/黄鹤楼商业库存.csv',engine='python')

df = pd.merge(df, df_store, left_on=['city_name','bar_name'],right_on=[u'地市',u'规格代码'], how='left')
df = df.rename(columns={'商业库存/箱': 'store_num'})

#print(df.columns)
df = df.drop(df.columns[[1, 3,4,5,6,9,10,11,12,13]], axis=1)
#print(df.columns)

df['city_bar_name']=df['city_name'].astype('str')+'_'+df['bar_name'].astype('str')
df_test = pd.get_dummies(df, columns=['city_bar_name'])
#print(df_test.columns)

df = df_test.drop(['city_name', 'bar_name'],axis=1)

x = df[df.columns[1:]]
y=df['sale_num']
#print(df.columns[1:])


import statsmodels.api as sm

#估计：就是模型
varLagNum = 1
df.index = df.month_id
model = sm.tsa.VARMAX(x,order=(varLagNum,0),trend='nc',exog=None)

fitMod = model.fit(maxiter=1000,disp=False)
# 打印统计结果
print(fitMod.summary())
# 获得模型残差
resid = fitMod.resid
result = {'fitMod':fitMod,'resid':resid}
exit()