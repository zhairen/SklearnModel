#-*- coding:utf-8 -*-
#1. 读取数据
import pandas as pd
df_read = pd.read_csv(r'DATA黄鹤楼系列_2016-2021年_销量数据.csv',engine='python')
#print(df)
df = df_read[df_read['sshengfen']=='湖北省']
#print(df.columns)
df = df.drop(['city_code','sshengfen','brand_code','brand_name','bar_code'], axis=1).drop_duplicates()
print('df',len(df))

df['sale_num'] =pd.to_numeric(df['sale_num'])

df_group = df.groupby(['month_id','city_name','bar_name'])['sale_num'].max().reset_index()
print('df_group',len(df_group))

#
df_store= pd.read_csv(r'黄鹤楼商业库存.csv',engine='python')
#print(df_store.columns)
df_store = df_store.drop(['地市代码','规格'], axis=1).drop_duplicates()
print('df_store',len(df_store))

df_store['商业库存/箱'] =df_store['商业库存/箱'].apply(lambda x: float(str(x).replace(',','')))

df_store_group = df_store.groupby([u'月份',u'地市',u'规格代码'])['商业库存/箱'].max().reset_index()
print('df_store_group',len(df_store_group))

df = pd.merge(df_group, df_store_group, left_on=['month_id','city_name','bar_name'],right_on=[u'月份',u'地市',u'规格代码'], how='left')
df = df.rename(columns={'商业库存/箱': 'store_num'})
print(len(df))
print(df.columns)

df = df.drop(['月份', '地市', '规格代码'], axis=1)
#print(df.columns)

df['city_bar_name']=df['city_name'].astype('str')+'_'+df['bar_name'].astype('str')
df_test = pd.get_dummies(df, columns=['city_bar_name'])

#df = df_test.drop(['city_name', 'bar_name'],axis=1)
df = df_test
#print(df.columns)

df.index = df.month_id

#时间序列
df_data = pd.DataFrame()
df_data['month_id']  = df['month_id'].drop_duplicates().copy()
df_data=df_data.drop(['month_id'],axis=1)
#df_data['test']='test'
#print(df_data)
import numpy as np

city_name = df.columns[5:6].str.split('_')[0][-2:-1][0]
bar_name = df.columns[5:6].str.split('_')[0][-1:][0]
print(city_name,bar_name)

df_filter = df[(df['city_name']==city_name) & (df['bar_name'] == bar_name)][df.columns[0:6]].copy()\
    .drop(['month_id'],axis=1)
df_data = pd.merge(df_data, df_filter, left_on=['month_id'],right_on=['month_id'], how='left')
#print(df_data.columns)

df_filter = df[df['city_name']==city_name][df.columns[0:6]].copy()\
    .drop(['month_id'],axis=1).groupby(['month_id'])['sale_num'].sum().reset_index(name='total_sale_num')
df_data = pd.merge(df_data, df_filter, left_on=['month_id'],right_on=['month_id'], how='left')
#print(df_data.columns)

df_filter = df[df['city_name']==city_name][df.columns[0:6]].copy()\
    .drop(['month_id'],axis=1).groupby(['month_id'])['store_num'].sum().reset_index(name='total_store_num')
df_data = pd.merge(df_data, df_filter, left_on=['month_id'],right_on=['month_id'], how='left')
print(df_data.columns)

df_data = df_data.drop(['city_name', 'bar_name'],axis=1).fillna(0)
print(df_data)

#df_data=df_data.set_index(['month_id'])

#2. train data, test data.
train_data = df_data[:len(df_data)-16]
test_data = df_data[len(df_data)-16:]

endog = train_data[['total_store_num','total_sale_num']]
exog = train_data[train_data.columns[1:2]]
print(endog.columns,exog.columns)

#估计：就是模型
import statsmodels.api as sm
from matplotlib import pyplot as plt

varLagNum = 12
model = sm.tsa.VARMAX(endog,order=(varLagNum,0),trend='n',exog=None)
res  = model.fit(maxiter=20, disp=False)
# 打印统计结果
print(res.summary())

y_fit = res.fittedvalues
#print('fitted values ',y_fit)

figure,ax=plt.subplots(2,2)
figure.autofmt_xdate()

from datetime import datetime
import matplotlib.dates as mdate

xdate = train_data['month_id'].apply(lambda  x:datetime.strptime(str(x),'%Y%m'))
ax[0][0].xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
#print(xdate[0],xdate)

ax[0][0].plot(xdate,train_data['total_sale_num'],'o', label='train_data')
ax[0][0].plot(xdate,y_fit['total_sale_num'],'r-', label='model_fit_data')
ax[0][1].xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
ax[0][1].plot(xdate,train_data['total_store_num'],'o', label='train_data')
ax[0][1].plot(xdate,y_fit['total_store_num'],'r-', label='model_fit_data')

ax[1][0].xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
ax[1][0].plot(xdate,y_fit['total_sale_num'],'r-', label='model_fit_data')
ax[1][1].xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
ax[1][1].plot(xdate,y_fit['total_store_num'],'r-', label='model_fit_data')

figure.suptitle('City : '+city_name)
ax[0][0].set_title('销售量')
ax[0][1].set_title('库存量')
ax[1][0].set_title('模型估计销售量')
ax[1][1].set_title('模型估计库存量')

ax[0][0].legend()
ax[0][1].legend()
ax[1][0].legend()
ax[1][1].legend()

plt.show()

df_res =pd.DataFrame(res.params[:len(res.params)-3])
df_res.to_csv('res_params.csv')

df_res['params']=df_res.index
df_res_store = df_res[df_res['params'].str.contains(".total_store_num.total_sale_num")]
df_res_store.to_csv('res_params_store.csv')
df_res_sale = df_res[df_res['params'].str.contains(".total_sale_num.total_sale_num")]
df_res_store.to_csv('res_params_sale.csv')

#print(len(train_data),train_data[['total_store_num','total_sale_num']])

y_predict = res.predict(start=48,end=51)['total_sale_num']
#print(y_predict)
y_test  = test_data['total_sale_num'][0:4]

xtest_date = test_data['month_id'].apply(lambda  x:datetime.strptime(str(x),'%Y%m'))[0:4]

ax=plt.gca()
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
#print(xdate.append(xtest_date))
plt.plot(xdate.append(xtest_date),train_data['total_sale_num'].append(y_test),'o', label='test_data')
plt.plot(xdate,y_fit['total_sale_num'],'r-', label='var_data')
#print(xdate[0],y_fit['total_sale_num'])
plt.plot(xdate.append(xtest_date)[-len(xtest_date)-1:],y_fit['total_sale_num']\
         .append(y_predict)[-len(y_predict)-1:],'r--', label='predict_data')
plt.legend()
plt.show()

imp_res = res.impulse_responses(10, orthogonalized=True)

#https://blog.csdn.net/Imliao/article/details/80352158

#具体烟型号的影响
new_model = sm.tsa.VARMAX(endog,order=(varLagNum,0),trend='n',exog=exog)
new_res  = new_model.fit(maxiter=20, disp=False)
#print(new_res.summary())
new_y_fit = new_res.fittedvalues

new_y_predict = new_res.predict(exog=[[100],[200],[2000],[300]],start=48,end=51)['total_sale_num']#exog[-4:]
print(y_predict)
print(new_y_predict)
ax=plt.gca()
ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
plt.plot(xdate.append(xtest_date),train_data['total_sale_num'].append(y_test),'o', label='test_data')
plt.plot(xdate,y_fit['total_sale_num'],'r-', label='var_data')
plt.plot(xdate.append(xtest_date)[-len(xtest_date)-1:],y_fit['total_sale_num']\
         .append(y_predict)[-len(y_predict)-1:],'r--', label='predict_data')
plt.plot(xdate.append(xtest_date)[-len(xtest_date)-1:],new_y_fit['total_sale_num']\
         .append(new_y_predict)[-len(new_y_predict)-1:],'b--', label='new_predict_data')
plt.legend()
plt.show()