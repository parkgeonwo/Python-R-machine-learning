# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 21:10:56 2021

@author: parkk
"""

# ▩ 보스톤 하우징 데이터의 집값을 예측하는 앙상블 모델 만들기

# 예제 1 . 보스톤 하우징 선형회귀 모델의 상관계수값과 오버피팅 여부확인

# 1. 사이킷런의 보스톤 데이터를 로드합니다.

from sklearn import datasets

boston = datasets.load_boston()
# print(boston)

x = boston.data
y = boston.target

# 2. 훈련 데이터와 테스트 데이터로 분리합니다. ( 9:1 )

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x,y, test_size = 0.1, random_state = 1 )
print( x_train.shape )              # 455 , 13
print( x_test.shape )                # 51 , 13

# 3. 선형회귀 모델을 생성합니다.

from sklearn.linear_model import LinearRegression

r1 = LinearRegression()
train_result1 = r1.fit( x_train, y_train ).predict(x_train)      # 훈련 데이터에 대한 예측
test_result1 = r1.fit( x_train, y_train ).predict(x_test)       # 테스트 데이터에 대한 예측

# 4. 상관계수값을 확인합니다.

print( np.corrcoef( y_train, train_result1 ) )          # 0.857
print( np.corrcoef( y_test, test_result1 ) )              # 0.889

※ 설명 : 오버피팅이 발생하지는 않았지만 성능이 좋지 않습니다.

# 예제 2 . 보스톤 하우징 랜덤포레스트 모델의 상관계수값과 오버피팅 여부확인

# 1. 사이킷런의 보스톤 데이터를 로드합니다.

from sklearn import datasets

boston = datasets.load_boston()
# print(boston)

x = boston.data
y = boston.target

# 2. 훈련 데이터와 테스트 데이터로 분리합니다. ( 9:1 )

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x,y, test_size = 0.1, random_state = 1 )
print( x_train.shape )              # 455 , 13
print( x_test.shape )                # 51 , 13

# 3. 선형회귀 모델을 생성합니다.

from sklearn.ensemble import RandomForestRegressor

r1 = RandomForestRegressor( random_state = 1 )
train_result2 = r1.fit( x_train, y_train ).predict(x_train)      # 훈련 데이터에 대한 예측
test_result2 = r1.fit( x_train, y_train ).predict(x_test)       # 테스트 데이터에 대한 예측

# 4. 상관계수값을 확인합니다.

print( np.corrcoef( y_train, train_result2 ) )          # 0.992
print( np.corrcoef( y_test, test_result2 ) )              # 0.967

# ※ 설명 : 회귀분석일때 보다는 정확도가 더 높아졌습니다. 그런데 회귀분석일 때와 마찬가지로
# 		훈련 데이터와 테스트 데이터의 성능차이가 발생하고 있습니다.
# 		랜덤포레스트의 경우는 오버피팅이 발생하고 있습니다.
		
# 예제 3 . 회귀모형과 랜덤포레스트 모형을 결합해서 앙상블 모형을 만들고 수치예측을 하시오 

# 1. 사이킷런의 보스톤 데이터를 로드합니다.

from sklearn import datasets

boston = datasets.load_boston()
# print(boston)

x = boston.data
y = boston.target

# 2. 훈련 데이터와 테스트 데이터로 분리합니다. ( 9:1 )

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x,y, test_size = 0.1, random_state = 1 )
print( x_train.shape )              # 455 , 13
print( x_test.shape )                # 51 , 13

# 3. 모델 2개를 결합한 앙상블 모형을 만들고 예측한다.

import numpy as np
from sklearn.linear_model import LinearRegression          # 선형회귀 모델 모듈 임폴트
from sklearn.ensemble import RandomForestRegressor   # 랜덤포레스트 모델 모듈 임폴트
from sklearn.ensemble import VotingRegressor          # 앙상블의 수치예측한 결과를 평균내는 모듈

r1 = LinearRegression()          # 선형회귀 모델 생성
r2 = RandomForestRegressor(n_estimators=10, random_state=1)        # 랜덤 포레스트 모델 생성

er = VotingRegressor([('lr', r1), ('rf', r2)])         # 선형회귀와 랜덤포레스트를 결합한 강력한 앙상블 모델 ER 생성

train_result3 = er.fit( x_train, y_train ).predict(x_train)      # 훈련 데이터에 대한 예측
test_result3 = er.fit( x_train, y_train ).predict(x_test)       # 테스트 데이터에 대한 예측

# 4. 상관계수값을 확인합니다.

print( np.corrcoef( y_train, train_result3 ) )          # 0.958
print( np.corrcoef( y_test, test_result3 ) )              # 0.956

# ※ 설명 : 앙상블 모형으로 만들었더니 훈련 데이터와 테스트 데이터간의 성능차이가 비슷하게 나온다.

# 	여기서 상관계수값을 더 올리는 것은 파생변수 추가, 10-hold 교차검정, 이상치 제거,
# 	결측치 대치를 통해서 더 올리면 됩니다.











