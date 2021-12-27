# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 21:13:21 2021

@author: parkk
"""

# 나이브 베이즈 모형을 이용해서 유방암 데이터 분류하기

# 1. 데이터를 로드합니다.

from sklearn import datasets
breast = datasets.load_breast_cancer()
x = breast.data
y = breast.target

# 2. 훈련 데이터와 테스트 데이터를 분리합니다.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size =0.1, random_state =1 )

# 3. 나이브베이즈 모델을 생성합니다.

from sklearn.naive_bayes import GaussianNB

r1 = GaussianNB()

# 4. 훈련/테스트 데이터 각각 예측

train_result = r1.fit( x_train, y_train ).predict(x_train)
test_result = r1.fit( x_train, y_train ).predict(x_test)

# 5. 훈련 데이트의 정확도와 테스트 데이터의 정확도를 확인합니다.

print ( sum( train_result == y_train ) / len( y_train ) )       # 0.939453125
print( sum( test_result == y_test ) / len( y_test ) )          # 0.9473684210526315

# ※ 정확도는 0.93 과 0.94로 차이가 살짝 발생한다.

#  유방암 데이터를 로지스틱 회귀 모델로 생성하고 예측하기

# 1. 데이터를 로드합니다.

from sklearn import datasets
breast = datasets.load_breast_cancer()
x = breast.data
y = breast.target

# 2. 훈련 데이터와 테스트 데이터를 분리합니다.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size =0.1, random_state =1 )

# 3. 로지스틱 회귀 모델을 생성합니다.

from sklearn.linear_model import LogisticRegression

r2 = LogisticRegression()

# 4. 훈련/테스트 데이터 각각 예측

train_result2 = r2.fit( x_train, y_train ).predict(x_train)
test_result2 = r2.fit( x_train, y_train ).predict(x_test)

# 5. 훈련 데이트의 정확도와 테스트 데이터의 정확도를 확인합니다.

print ( sum( train_result2 == y_train ) / len( y_train ) )       # 0.943359375
print( sum( test_result2 == y_test ) / len( y_test ) )          # 1

# ※ 설명 : 로지스틱 회귀모델은 나이브 베이즈 보다 정확도는 높으나 정확도의 차이가 심합니다.

# 유방암 데이터를 분류하는 머신러닝 모델을 랜덤포레스트 모델로 생성하고 분류하시오

# 1. 데이터를 로드합니다.

from sklearn import datasets
breast = datasets.load_breast_cancer()
x = breast.data
y = breast.target

# 2. 훈련 데이터와 테스트 데이터를 분리합니다.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size =0.1, random_state =1 )

# 3. 랜덤포레스트 모델을 생성합니다.

from sklearn.ensemble import RandomForestClassifier

r3 = RandomForestClassifier()

# 4. 훈련/테스트 데이터 각각 예측

train_result3 = r3.fit( x_train, y_train ).predict(x_train)
test_result3 = r3.fit( x_train, y_train ).predict(x_test)

# 5. 훈련 데이트의 정확도와 테스트 데이터의 정확도를 확인합니다.

print ( sum( train_result3 == y_train ) / len( y_train ) )       # 1
print( sum( test_result3 == y_test ) / len( y_test ) )          # 0.9473684210526315

# ※ 설명 : 랜덤포레스트 모델은 정확도는 아주 높으나 오버피팅이 발생했습니다.

# 예제 4. ( 점심시간 문제 ) 위의 3개의 모델을 결합한 강력한 앙상블 모델을 만드시오 !

# ■ 변경 1.
from sklearn.ensemble import VotingRegressor   ( 수치예측 )
			↓
from sklearn.ensemble import VotingClassifier  ( 분류 )

# ■ 변경 2.
er = VotingRegressor( [ ( 'lr', r1 ) , ( 'rf', r2 ) ] )
			↓
eclf1 = VotingClassifier( estimators = [ ( 'gnb', r1 ) , ('lr', r2) , ('rf', r3) ] , voting = 'hard'  )


# 1. 데이터를 로드합니다.

from sklearn import datasets
breast = datasets.load_breast_cancer()
x = breast.data
y = breast.target

# 2. 훈련 데이터와 테스트 데이터를 분리합니다.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size =0.1, random_state =1 )

# 3. 모델 3개를 결합한 앙상블 모형을 만들고 예측한다.

import numpy as np
from sklearn.naive_bayes import GaussianNB     # 나이브베이즈 모델 모듈 임폴트
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀 모델 모듈 임폴트
from sklearn.ensemble import RandomForestClassifier   # 랜덤포레스트 분류 모델 모듈 임폴트

from sklearn.ensemble import VotingClassifier     # 앙상블의 분류 결과를 평균내는 모듈

r1 = GaussianNB()          # 나이브 베이즈 모델 생성
r2 = LogisticRegression()      # 로지스틱 회귀 모델 생성
r3 = RandomForestClassifier()  # 랜덤포레스트 모델 생성

eclf1 = VotingClassifier( estimators = [ ( 'gnb', r1 ) , ('lr', r2) , ('rf', r3) ] , voting = 'hard'  )
        # 나이브베이즈/ 로지스틱회귀/ 랜덤포레스트  결합한 강력한 앙상블 모델 eclf1 생성

train_result4 = eclf1.fit( x_train, y_train ).predict(x_train)      # 훈련 데이터에 대한 예측
test_result4 = eclf1.fit( x_train, y_train ).predict(x_test)       # 테스트 데이터에 대한 예측

# 4. 훈련 데이트의 정확도와 테스트 데이터의 정확도를 확인합니다.

print ( sum( train_result4 == y_train ) / len( y_train ) )       # 0.970703125
print( sum( test_result4 == y_test ) / len( y_test ) )          # 0.9649122807017544





