# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 16:49:45 2021

@author: parkk
"""

hg = pd.read_csv("c:\\data\\simple_hg.csv")

x = hg[['cost']]         # 독립변수
y = hg[['input']]        # 종속변수

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x, y)

print('기울기 :', model.coef_ )
print('절편 :',model.intercept_ )

#
# 기울기 : [[2.18648985]]
# 절편 : [62.92913386]
# 매출액 = 2.1864*광고비 + 62.92913

result = model.predict([[26]])    
print(result)                                  # [[119.77786987]] , 11.9억

y_hat = model.predict(x)               # 회귀직선에 넣고 예측한값

import matplotlib.pyplot as plt       # 그래프 그리기위한 모듈
import seaborn as sns

plt.figure( figsize = (10,5) )               # 그래프 사이즈 가로 10, 세로 5
ax1 = sns.distplot( y, hist = False, label = 'y', color = 'red' )          # 실제 값을 라인그래프로 시각화
ax2 = sns.distplot( y_hat, hist= False, label = 'y_hat', ax = ax1, color ='blue' )  # 예측값을 라인그래프로 시각화
plt.show()
plt.close()




r_square = model.score( x,y )
print(r_square)                           # 0.7884035286357817

sns.regplot(x,y)







