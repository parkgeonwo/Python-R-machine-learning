# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 17:54:52 2021

@author: parkk
"""

# 1. 데이터를 로드합니다.

import pandas as pd
wine = pd.read_csv("c:\\data\\whitewines.csv")

# 2. 결측치를 확인합니다.

print( wine.isnull().sum() )

# 3. 종속변수의 정규성을 확인합니다.

wine['quality'].plot(kind = 'hist')

# 4. 훈련데이터와 테스트데이터를 분리합니다.

from sklearn.model_selection import train_test_split

x = wine.iloc[ :, :-1 ].to_numpy()
y = wine.iloc[ :, -1  ].to_numpy()

x_train, x_test, y_train, y_test = train_test_split ( x, y, test_size = 0.1 , random_state = 1 )

print(x_train.shape )     # 4408, 11
print(x_test.shape )       # 490, 11
print(y_train.shape )     # 4408,
print(y_test.shape )       # 490,

# 5. 모델을 생성합니다.

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor( random_state = 1 )

# 6. 모델 훈련

model.fit( x_train, y_train )

# 7. 모델 예측

result = model.predict( x_test )

# 8. 실제값과 예측값간의 상관계수와 오차를 확인합니다.

import numpy as np

print ( np.corrcoef( result, y_test ) )          # 0.58180875

def mae( x, y ):
    return np.mean( abs( x-y ) )

print(  mae( result, y_test )  )      # 0.47551020408163264

# ※ 설명 : 의사결정트리 회귀모델로 수치예측결과 상관계수는 0.58 , 오차는 0.47로 출력되었다.
# 	위의 수치예측 모델의 성능을 올리시오 !
		
from sklearn.tree import DecisionTreeRegressor
				↓
from sklearn.ensemble import RandomForestRegressor

# RandomForestRegressor 로 변경해서 수치예측하고 상관계수와 오차를 확인하시오 !

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor( random_state = 1 )

model.fit( x_train, y_train )

result = model.predict( x_test )

import numpy as np

print ( np.corrcoef( result, y_test ) )          # 0.71178998

def mae( x, y ):
    return np.mean( abs( x-y ) )

print(  mae( result, y_test )  )      # 0.43489795918367347









