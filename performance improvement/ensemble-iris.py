# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 21:19:48 2021

@author: parkk
"""

# 사이킷런의 아이리스 데이터로 앙상블 모델을 만들어서 분류하시오 

# 1. 데이터를 로드합니다.

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

# 2. 훈련 데이터와 테스트 데이터를 분리합니다.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size =0.1, random_state =1 , stratify = y )

# 3. 모델 3개를 결합한 앙상블 모형을 만들고 예측한다.

import numpy as np
from sklearn.naive_bayes import GaussianNB     # 나이브베이즈 모델 모듈 임폴트
from sklearn.neural_network import MLPClassifier  # 신경망 모델 모듈 임폴트
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀 모델 모듈 임폴트
from sklearn.ensemble import RandomForestClassifier   # 랜덤포레스트 분류 모델 모듈 임폴트
# from sklearn.pipeline import make_pipeline

from sklearn.ensemble import VotingClassifier     # 앙상블의 분류 결과를 평균내는 모듈

r0 = GaussianNB()          # 나이브 베이즈 모델 생성
r1 = MLPClassifier( random_state = 1 )
r2 = LogisticRegression( random_state = 1 )      # 로지스틱 회귀 모델 생성
r3 = RandomForestClassifier( random_state = 1 )  # 랜덤포레스트 모델 생성

eclf1 = VotingClassifier( estimators = [ ( 'gnb', r0 ),( 'mlp', r1 ) , ('lr', r2) , ('rf', r3) ] , voting = 'hard'  )
        # 나이브베이즈/신경망/ 로지스틱회귀/ 랜덤포레스트  결합한 강력한 앙상블 모델 eclf1 생성

# pipeline = make_pipeline( MinMaxScaler(), eclf1 )

train_result4 = eclf1.fit( x_train, y_train ).predict(x_train)      # 훈련 데이터에 대한 예측
test_result4 = eclf1.fit( x_train, y_train ).predict(x_test)       # 테스트 데이터에 대한 예측

# train_result4 = pipeline.fit( x_train, y_train ).predict(x_train)      # 훈련 데이터에 대한 예측
# test_result4 = pipeline.fit( x_train, y_train ).predict(x_test)       # 테스트 데이터에 대한 예측

# 4. 훈련 데이트의 정확도와 테스트 데이터의 정확도를 확인합니다.

print ( sum( train_result4 == y_train ) / len( y_train ) )       # 0.9851851851851852
print( sum( test_result4 == y_test ) / len( y_test ) )          # 0.9333333333333333


# ※ 설명 : test 데이터에 대한 정확도가 낮은 오버피팅이 일어난다.
# 		파이프라인 적용시 train 에 대한 정확도 는 0.9629629629629629, 
# 		test에 대한 정확도는 0.9333333333333333로서 오히려 정확도가 떨어졌다.












