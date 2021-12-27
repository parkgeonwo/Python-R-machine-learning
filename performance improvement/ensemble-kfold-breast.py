# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 21:18:25 2021

@author: parkk
"""

# ▩ k 홀드 교차검정으로 정확도 높이기

# 1. 데이터를 로드합니다.

from sklearn import datasets
breast = datasets.load_breast_cancer()
x = breast.data
y = breast.target

# 2. 훈련 데이터와 테스트 데이터를 분리합니다.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size =0.1, random_state =1 , stratify = y)

# 3. 모델 3개를 결합한 앙상블 모형을 만들고 예측한다.

import numpy as np
from sklearn.naive_bayes import GaussianNB     # 나이브베이즈 모델 모듈 임폴트
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀 모델 모듈 임폴트
from sklearn.ensemble import RandomForestClassifier   # 랜덤포레스트 분류 모델 모듈 임폴트

from sklearn.ensemble import VotingClassifier     # 앙상블의 분류 결과를 평균내는 모듈
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

# r1 = GaussianNB()          # 나이브 베이즈 모델 생성
# r2 = LogisticRegression()      # 로지스틱 회귀 모델 생성

# k홀드 교차검정을 10으로 설정

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold( n_splits = 10 )        # 10개의 fold를 만들어서 교차검정하겠다.
kfold = cv.split( x_train, y_train )      # 10개의 fold의 훈련데이터와 검정데이터를 kfold에 넣겠다.

# 모델생성

r3 = RandomForestClassifier( n_estimators = 10  , random_state = 1)  # 랜덤포레스트 모델 생성 / 하이퍼 파라미터

# eclf1 = VotingClassifier( estimators = [ ( 'gnb', r1 ) , ('lr', r2) , ('rf', r3) ] , voting = 'hard'  )
        # 나이브베이즈/ 로지스틱회귀/ 랜덤포레스트  결합한 강력한 앙상블 모델 eclf1 생성

pipeline = make_pipeline( MinMaxScaler(), r3 )   # eclf1 이란 모델에 정규화를 함께 묶어 pipeline 으로

# 4. 훈련데이터, 테스트 데이터 예측

score2 = []
for k, ( train, test ) in enumerate(kfold):
    # print( k )
    # print( train )
    # print( test )
    pipeline.fit( x_train[ train, : ], y_train[train] )      # 10 fold 데이터셋을 훈련시켜서
    score = pipeline.score( x_train[ test, : ], y_train[ test ] )   # 검정 데이터 10개에 대한 정확도가
    score2.append( score )   # score2 에 append 된다.
print( score2 )    # 정확도 10개가 출력
print( np.mean( score2 ) )            # 0.9628205128205127

						# 정확도 평균
# n_estimators = 10 , random_state = 1 --------> 0.9569381598793363
# n_estimators = 50 , random_state = 1 --------> 0.9628205128205127
# n_estimators = 100 , random_state = 1 ---------> 0.9647812971342382


# ※ 설명 : pipeline에 standardScaler와 kfold 를 같이 사용했는데 kfold 를 standardScaler와
# 		효율적으로 사용하려고 pipeline 을 사용한것입니다.






