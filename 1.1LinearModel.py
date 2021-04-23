#1.1.1普通最小二乘法
from sklearn import linear_model
reg = linear_model.LinearRegression()
x = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
reg.fit (x,y)
print('1.1.1 LeastSquare',reg.coef_)
    #画图
#import matplotlib.pyplot as plt
#plt.plot(x, y, 'b.')
#plt.plot(x, reg.predict(x), 'r')
#plt.xlabel('普通最小二乘法')
#plt.show()
#1.1.2岭回归

from sklearn import linear_model
reg = linear_model.Ridge (alpha = .1)
x=[[0, 0], [0, 0], [1, 1]]
y= [0, .1, 1]
reg.fit (x,y)

from sklearn import linear_model
reg2 = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
reg2.fit(x, y)
print('1.1.2 Ridge',reg.coef_,reg.intercept_,reg2.alpha_)


#1.1.3 Lasso
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso

'''创造数据X Y'''
reg_data, reg_target = make_regression(n_samples=200, n_features=10, n_informative=5, noise=5)

''' 通过交叉检验来获取最优参数'''
from sklearn.linear_model import LassoCV

lassocv = LassoCV()
lassocv.fit(reg_data, reg_target)
alpha = lassocv.alpha_
print('利用Lasso交叉检验计算得出的最优alpha：' + str(alpha))

'''lasso回归'''
lasso = Lasso(1)
lasso.fit(reg_data, reg_target)

'''计算系数不为0的个数'''
import numpy as np

n = np.sum(lasso.coef_ != 0)
print('Lasso回归后系数不为0的个数：' + str(n))

'''输出结果
   如果names没有定义，则用X1 X2 X3代替
   如果Sort = True，会将系数最大的X放在最前'''

def pretty_print_linear(coefs, names=None, sort=False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key=lambda x: -np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                      for coef, name in lst)

print('Y = ' + pretty_print_linear(lasso.coef_))

#1.1.3 Lasso https://www.freesion.com/article/4698918476/
# -*- coding: utf-8 -*-

"""
Lasso 回归应用于稀疏信号
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# 用于产生稀疏数据
np.random.seed(int(time.time()))
# 生成系数数据，样本为50个，参数为200维
n_samples, n_features = 50, 200
# 基于高斯函数生成数据
X = np.random.randn(n_samples, n_features)
# 每个变量对应的系数
coef = 3 * np.random.randn(n_features)
# 变量的下标
inds = np.arange(n_features)
# 变量下标随机排列
np.random.shuffle(inds)
# 仅仅保留10个变量的系数，其他系数全部设置为0
# 生成稀疏参数
coef[inds[10:]] = 0
# 得到目标值，y
y = np.dot(X, coef)
# 为y添加噪声
y += 0.01 * np.random.normal((n_samples,))

# 将数据分为训练集和测试集
n_samples = X.shape[0]
X_train, y_train = X[:int(n_samples / 2)], y[:int(n_samples / 2)]
X_test, y_test = X[int(n_samples / 2):], y[int(n_samples / 2):]

# Lasso 回归的参数
alpha = 0.1
lasso = Lasso(max_iter=10000, alpha=alpha)

# 基于训练数据，得到的模型的测试结果
# 这里使用的是坐标轴下降算法（coordinate descent）
y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)

# 这里是R2可决系数（coefficient of determination）
# 回归平方和（RSS）在总变差（TSS）中所占的比重称为可决系数
# 可决系数可以作为综合度量回归模型对样本观测值拟合优度的度量指标。
# 可决系数越大，说明在总变差中由模型作出了解释的部分占的比重越大，模型拟合优度越好。
# 反之可决系数小，说明模型对样本观测值的拟合程度越差。
# R2可决系数最好的效果是1。
r2_score_lasso = r2_score(y_test, y_pred_lasso)  # 实际的y，与lasso预测的y

print("测试集上的R2可决系数 : %f" % r2_score_lasso)

lasso.fit(X_train, y_train)

figure,ax=plt.subplots(1,3)
ax[0].plot(lasso.coef_, label='Lasso coefficients')
ax[0].plot(coef, '--', label='original coefficients')
ax[0].legend(loc='best')
ax[0].set_title('train sample')

lasso.fit(X_test, y_test)
ax[1].plot(lasso.coef_, label='Lasso coefficients')
ax[1].plot(coef, '--', label='original coefficients')
ax[1].legend(loc='best')
ax[1].set_title('test sample')

lasso.fit(X, y)
ax[2].plot(lasso.coef_, label='Lasso coefficients')
ax[2].plot(coef, '--', label='original coefficients')
ax[2].legend(loc='best')
ax[2].set_title('all sample')
#plt.show()

#Pipelines: chaining pre-processors and estimators
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create a pipeline object
pipe = make_pipeline(
StandardScaler(),
LogisticRegression()
)

# load the iris dataset and split it into train and test sets
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# fit the whole pipeline
pipe.fit(X_train, y_train)

# we can now use it like any other estimator
print('accuracy_score ',accuracy_score(pipe.predict(X_test), y_test))

#Model evaluation
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

X, y = make_regression(n_samples=1000, random_state=0)
lr = LinearRegression()
result = cross_validate(lr, X, y)  # defaults to 5-fold CV
print(result['test_score'])

#Automatic parameter searches
