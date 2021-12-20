# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 21:00:17 2021

@author: parkk
"""

# ▩ 파이썬으로 콘크리트 강도를 예측하는 인공신경망 구현하기

# 1. 데이터를 로드합니다.
import pandas as pd
df = pd.read_csv("c:\\data\\concrete.csv")
print(df.shape)                # (1030, 9)

# 2. 결측치를 확인합니다.
print(df.isnull().sum())

# 3. 이상치를 확인합니다.

def outlier_value(x):

    for i in list(x.describe().columns):      # x.columns[(x.dtypes =='float64') | (x.dtypes == 'int64')] ,  x.columns[x.dtypes.isin(['float64','int64']) 
        Q1 = x[i].quantile(0.25)
        Q3 = x[i].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + (IQR*1.5)
        lower_bound = Q1 - (IQR*1.5)
        a = x.loc[ (x[i] > upper_bound ) | ( x[i]< lower_bound ) , i ].count()          # 이상치의 건수
        b = i
        print( '{0:<10} : {1:>5} 건'.format(  b , a  ) )

outlier_value( df )


# slag       :     2 건
# water      :     9 건
# superplastic :    10 건
# fineagg    :     5 건
# age        :    59 건
# strength   :     4 건

# 4. 정규화를 진행합니다.

x = df.iloc[ :, 0:-1 ].to_numpy()
y = df['strength']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x)

x2 = scaler.transform(x)

y2 = y.to_numpy()

# 5. 훈련 데이터와 테스트 데이터를 분리합니다.

from sklearn.model_selection import train_test_split

x_train , x_test, y_train, y_test = train_test_split ( x2, y2, test_size = 0.2, random_state = 1 )

print( x_train.shape )          # 824, 8
print( x_test.shape )            # 206, 8
print( y_train.shape )           # 824
print( y_test.shape )             # 206

# 6. 모델 생성

from sklearn.neural_network import MLPRegressor

model = MLPRegressor( random_state = 0 )

# 7. 모델 훈련

model.fit( x_train, y_train )

# 8. 모델 예측

result = model.predict(x_test)

# 9. 모델 평가 ( 상관계수 확인 )

import scipy.stats as stats

print( stats.pearsonr( y_test, result ) )     # (0.76397431733998, 1.1119309961387144e-40)

# 10. 모델 성능 개선

# 모델 생성 부분부터 다시 합니다.

from sklearn.neural_network import MLPRegressor

model2 = MLPRegressor( random_state = 0 , hidden_layer_sizes = ( 200, 50 )  )

# 설명 : 은칙1층의 뉴런의 갯수를 200개로 늘립니다. ( 기본값이 100 )
# 은닉 2층의 뉴런의 갯수를 50개로 늘립니다.
# 입력층 (0층) ---------> 은닉1층 -------> 은닉2층 ------> 출력층(3층)

model2.fit( x_train, y_train )

result2 = model2.predict(x_test)

import scipy.stats as stats

print( stats.pearsonr( y_test, result2 ) )     # (0.8335957259787699, 1.739638449302061e-54)

# hidden_layer_sizes 하이퍼 파라미터를 더 조절해서 모델의 성능을 더 올리시오

from sklearn.neural_network import MLPRegressor

model3 = MLPRegressor( random_state = 0 , hidden_layer_sizes = ( 200, 50,50 )  )

model3.fit( x_train, y_train )

result3 = model3.predict(x_test)

import scipy.stats as stats

print( stats.pearsonr( y_test, result3 ) )     # (0.8867328664497545, 2.7497293354463045e-70)












