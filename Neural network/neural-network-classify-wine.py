# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:38:11 2021

@author: parkk
"""

# ▩ 파이썬으로 분류하는 인공신경망 만들기

# 수치예측 : from sklearn.neural_network import MLPRegressor

# 분류 : from sklearn.neural_network import MLPClassifier

# wine의 품질을 분류하는 인공신경망을 파이썬으로 구현하시오 !

# 1. 데이터를 로드합니다.
import pandas as pd
wine = pd.read_csv("c:\\data\\wine.csv")

# 2. 결측치를 확인합니다.

print( wine.isnull().sum() )

# 3. 이상치를 확인합니다.

def outlier_value(x):

    for i in list(x.describe().columns):             # x.columns[(x.dtypes =='float64') | (x.dtypes == 'int64')] ,  x.columns[x.dtypes.isin(['float64','int64']) 
        Q1 = x[i].quantile(0.25)
        Q3 = x[i].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + (IQR*1.5)
        lower_bound = Q1 - (IQR*1.5)
        a = x.loc[ (x[i] > upper_bound ) | ( x[i]< lower_bound ) , i ].count()          # 이상치의 건수
        b = i
        print( '{0:<10} : {1:>5} 건'.format(  b , a  ) )

outlier_value( wine )

# 4. 정규화를 진행합니다.

x = wine.iloc[ : , 1: ]
y = wine.iloc[ : , 0 ]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x)
x2 = scaler.transform(x)
y2 = y.to_numpy()

# 5. 훈련 데이터와 테스트 데이터를 분리합니다.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split ( x2, y2, test_size = 0.1, random_state = 0 )

print( x_train.shape )      # 160, 13
print( x_test.shape )        # 18 ,13
print( y_train.shape )       # 160
print( y_test.shape )         # 18

# 6. 모델 생성

from sklearn.neural_network import MLPClassifier

model = MLPClassifier( random_state = 0 )

# 7. 모델 훈련

model.fit(x_train, y_train)

# 8. 모델 예측

result = model.predict( x_test )

# 9. 모델 평가

print ( sum( result == y_test ) / len(y_test) )                  # 1

# 10. 성능 개선

model = MLPClassifier( hidden_layer_sizes = (100,100) , activation = 'relu', solver = 'adam', random_state = 0 )




