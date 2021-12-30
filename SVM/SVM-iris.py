# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 13:12:50 2021

@author: parkk
"""

# ■ 파이썬으로 iris 데이터의 서포트 벡터 머신 모델 생성하기

# 1. 데이터 로드
import pandas as pd
iris = pd.read_csv("c:\\data\\iris2.csv")

# 2. min/max 정규화하기
x = iris.iloc[ : , 0:4 ]
y = iris[ 'Species' ]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x)

x_scaled = scaler.transform(x)

# 3. 훈련/테스트 데이터 분리
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x_scaled, y, test_size = 0.2, random_state = 1 )

# 4. 모델 생성하기
from sklearn import svm

svm_model = svm.SVC( kernel = 'rbf' )
# 커널의 종류 : rbf, poly, sigmoid, linear

#5. 모델 훈련
svm_model.fit( x_train, y_train )

# 6. 모델 예측
result = svm_model.predict(x_test)

# 7. 모델 평가
print ( sum( result == y_test ) / len(y_test) )         # 0.966667

# gridsearch 의 자동튜닝 기능을 이용해서 위의 아이리스 모델의 성능을 더 올리시오 !

# 힌트 :

# 모델 생성
from sklearn import svm
from sklearn.model_selection import GridSearchCV

param_grid = { 'C' : [ 0.1, 1, 10, 100, 1000 ],
			'gamma' : [ 1, 0.1, 0.01, 0.001, 0.0001 ],
			'kernel' : [ 'rbf', 'poly', 'sigmoid', 'linear' ] }

grid = GridSearchCV( svm.SVC() , param_grid, refit = True, cv = 3 , n_jobs = -1, verbose =2 )

# 설명 : refit = True 는 최적의 하이퍼 파라미터를 찾은 뒤 찾아낸 최적의 하이퍼 파라미터로 재학습시킨다.

# 모델 훈련
grid.fit( x_train, y_train )
print( grid.best_params_ )                   # {'C': 10, 'gamma': 1, 'kernel': 'linear'}

# 6. 모델 예측
result = grid.predict(x_test)

# 7. 모델 평가
print ( sum( result == y_test ) / len(y_test) )         # 0.966667 / 똑같이나옴

# 아래의 3개의 머신러닝 모델의 조합으로 앙상블 모델을 만들어서 아이리스 품종을 분류하는
# 머신러닝 모델을 만들고 정확도를 출력하시오 !
# 모델 1: 서포트 벡터 머신 / 모델 2: 나이브 베이즈 / 모델 3 : 랜덤포레스트
		
# 1. 데이터 로드
import pandas as pd
iris = pd.read_csv("c:\\data\\iris2.csv")

# 2. min/max 정규화하기
x = iris.iloc[ : , 0:4 ]
y = iris[ 'Species' ]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x)

x_scaled = scaler.transform(x)

# 3. 훈련/테스트 데이터 분리
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x_scaled, y, test_size = 0.2, random_state = 1 )


#3. 앙상블 모델생성
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier    #분류

r1=SVC()
r2=GaussianNB()
r3=RandomForestClassifier()

eclf1=VotingClassifier(estimators=[ ('svc', r1), ('gnb', r2), ('rf', r3) ], voting='hard')

#4. 훈련데이터, 테스트데이터 예측
train_result=eclf1.fit(x_train, y_train).predict(x_train)
test_result=eclf1.fit(x_train, y_train).predict(x_test)


#5. 정확도 확인
print(sum(train_result==y_train)/len(y_train))    # 0.966667
print(sum(test_result==y_test)/len(y_test))     # 0.966667











