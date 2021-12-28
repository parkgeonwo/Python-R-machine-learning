# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 21:19:48 2021

@author: parkk
"""

# ▩ 배깅 실습 ( 독일 은행 데이터의 채무 불이행자를 예측하는 모델 만들기 )

# 	1. 하나의 의사결정트리 모델로 구현

# 	2. 의사결정트리 + 배깅 모델로 구현

# ▩ 하나의 의사결정트리 모델로 구현

# 1. 데이터 로드

import pandas as pd
credit = pd.read_csv("c:\\data\\credit.csv")

# 2. 명목형 데이터를 더미변수화 한다.

credit2 = pd.get_dummies( credit.iloc[ : , :-1 ] )

# 3. 훈련 데이터와 테스트 데이터로 분리합니다.

x = credit2.to_numpy()
y = credit.iloc[ : , -1 ].to_numpy()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x, y , test_size = 0.1, random_state = 1 )

# 4. 정규화 작업 수행

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train2 = scaler.transform( x_train )
x_test2 = scaler.transform( x_test )

# 5. 의사결정트리 모델 생성

from sklearn.tree import DecisionTreeClassifier

model1 = DecisionTreeClassifier( criterion = "entropy", max_depth = 20, random_state = 1 )

# 6. 모델 훈련

model1.fit( x_train2, y_train )

# 7. 모델 예측

result1 = model1.predict(  x_test2 )

# 8. 모델 평가

print( sum( result1 == y_test ) / len(y_test) )        # 0.72


# ▩ 의사결정트리 + 배깅 모델로 구현

# 앞에 데이터 불러오는 부분은 이미 했으므로 그대로 두고 모델생성부터 시작

from sklearn.tree import DecisionTreeClassifier

model2 = DecisionTreeClassifier( criterion = 'entropy', max_depth = 20, random_state =1 )

from sklearn.ensemble import BaggingClassifier

bagging2 = BaggingClassifier( model2, max_samples = 0.9, max_features = 0.5, random_state = 1 )

# 설명 : max_samples = 0.9 는 bag 에 데이터 담을때 훈련 데이터의 90% 를 샘플링하겠다.
#             max_features = 0.5 하나의 예측기가 가져갈 수 있는 최대 컬럼의 갯수

# 모델훈련
bagging2.fit( x_train2, y_train )

# 모델 예측
result2 = bagging2.predict( x_test2 )

#모델평가
print( sum(result2 == y_test) / len(y_test) )         # 0.74













