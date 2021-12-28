# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 22:05:56 2021

@author: parkk
"""


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

# 9. Adaboosting 구현

from sklearn.ensemble import AdaBoostClassifier

model5 = AdaBoostClassifier( n_estimators = 100, random_state = 1 )

model5.fit( x_train2, y_train )      # 모델 훈련
result5 = model5.predict( x_test2 )  # 모델 예측
print( sum( result5 == y_test ) / len( y_test ) )      # 모델 평가  / 0.77

# 10. 그레디언트 부스팅으로 모델 생성

from sklearn.ensemble import GradientBoostingClassifier

model6 = GradientBoostingClassifier( n_estimators = 300, random_state = 1 )
model6.fit( x_train2, y_train )      # 모델 훈련
result6 = model6.predict( x_test2 )  # 모델 예측
print( sum( result6 == y_test ) / len( y_test ) )      # 모델 평가  / 0.78

# 11. xgboost 모델 생성

# 아나콘다 prompt 창 : pip install xgboost

from xgboost import XGBClassifier

model7 = XGBClassifier(  n_estimators = 300, random_state = 1 )
evals = [ ( x_test2, y_test ) ]
model7.fit( x_train2, y_train, early_stopping_rounds  = 100,
		eval_metric = 'logloss', eval_set = evals, verbose = 1 )      # 모델 훈련

# 설명 : early_stopping_rounds 은 과적합을 방지 시키기 위해서 학습을 조기종료시키는 기능
		# eval_metric 에 오차함수명을 기술하면 되는데 logloss 여러개의 오차함수들중에 하나입니다.
		# 오차함수명 mse, mae, logloss, error, merror, auc

result7 = model7.predict( x_test2 )  # 모델 예측
print( sum( result7 == y_test ) / len( y_test ) )      # 모델 평가  / 0.75









