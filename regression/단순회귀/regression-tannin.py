# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 16:24:07 2021

@author: parkk
"""

# 	" 탄닌 함유량과 애벌래 성장간의 관계에 대한 회귀식을 도출하기"
	
# 1. 데이터를 로드합니다.
# 2. 종속변수와 독립변수를 지정합니다.
# 3. 모델을 설정합니다.
# 4. 모델을 훈련시킵니다.
# 5. 기울기와 절편을 구합니다.


# 1. 데이터를 로드합니다.

import pandas 
reg = pd.read_csv("c:\\data\\regression.txt", sep = '\t')

# 2. 종속변수와 독립변수를 지정합니다.

x = reg[['tannin']]         # 독립변수
y = reg[['growth']]        # 종속변수

# 3. 모델을 설정합니다.

from sklearn.linear_model import LinearRegression

model = LinearRegression()

# 4. 모델을 훈련시킵니다.

model.fit(x, y)

# 5. 기울기와 절편을 구합니다.

print('기울기 :', model.coef_ )
print('절편 :',model.intercept_ )

#
# 기울기 : [[-1.21666667]]
# 절편 : [11.75555556]
# 성장률 = -1.216*탄닌함유량 + 11.755

# 6. 탄닌 함유량이 9일 때의 성장률을 예측하시오

result = model.predict([[9]])    
print(result)                                  # [[0.80555556]]

# 7. 위의 회귀 직선을 시각화 하시오 !

y_hat = model.predict(x)               # 회귀직선에 넣고 예측한값

import matplotlib.pyplot as plt       # 그래프 그리기위한 모듈
import seaborn as sns

plt.figure( figsize = (10,5) )               # 그래프 사이즈 가로 10, 세로 5
ax1 = sns.distplot( y, hist = False, label = 'y', color = 'red' )          # 실제 값을 라인그래프로 시각화
ax2 = sns.distplot( y_hat, hist= False, label = 'y_hat', ax = ax1, color ='blue' )  # 예측값을 라인그래프로 시각화
plt.show()
plt.close()



# *실제값과 예측값이 얼마나 일치하는지를 시각화한것

# 8. 훈련데이터를 얼마나 잘 설명하는지를 나타내는 지표인 결정계수값을 출력하시오 !
	( 1에 가까울 수록 데이터에 대한 설명력이 높습니다. )

r_square = model.score( x,y )
print(r_square)                           # 0.8156632653061224
