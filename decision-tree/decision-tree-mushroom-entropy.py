# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:05:07 2021

@author: parkk
"""

# 1 . 데이터를 로드합니다.

import pandas as pd
mush = pd.read_csv("c:\\data\\mushrooms.csv")
print ( mush.shape )      # (8124, 23)    # 데이터가 크니까 8대2로 나눠서 해도 되겠다.

# 2. 결측치를 확인합니다.

print( mush.isnull().sum() )        # 0

# ※ 설명 : 결측치가 있다면 그 컬럼의 중앙값, 최빈값, 평균값 등으로 대치합니다.

# 3. 이상치를 확인합니다.

# 명목형 데이터이기 때문에 이상치를 확인할 수 없다.

# ※ 설명 : 이상치가 있다면 이상치를 중앙값, 최빈값, 평균값 등으로 치환합니다.

# 4. 명목형 데이터가 있는지 확인하고 숫자형으로 변경합니다. ( R 과 다른점이 이부분입니다. )

print(mush.head())

mush2 = pd.get_dummies( mush.iloc [ :, 1: ]   )            # 정답 컬럼제외하고 숫자형으로 변경
print( mush2.head() )
x = mush2.to_numpy()                      # 숫자로 변경한 훈련데이터를 numpy array로 변경

y = mush['type'].to_numpy()           # 라벨데이터 생성 / numpy array로 바꿔줌

# 5. 훈련데이터와 테스트 데이터를 나눕니다. ( 8대2 )

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split ( x, y , test_size = 0.2, random_state = 1 )  # x,y 를 numpy array 형태로 받는다.

print(x_train.shape)        # (6499, 117)      / 훈련 데이터
print(x_test.shape)          # (1625, 117)     / 테스트 데이터
print(y_train.shape)         # (6499,)           / 훈련데이터의 라벨
print(y_test.shape)        # (1625,)             / 테스트 데이터의 라벨

# 6. 훈련데이터를 정규화 합니다.

# 전부 0과 1이므로 정규화 작업을 생략합니다.

# 7. 테스트 데이터를 정규화 합니다.

# 전부 0과 1이므로 정규화 작업을 생략합니다.

# 8. 의사결정트리 모델을 생성합니다.

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier ( criterion = 'entropy', max_depth = 5 )       #  max_depth : 가지의 깊이

# ※ 설명 : criterion 은 entropy 와 gini 가 있습니다.
# 		max_depth 는 가지의 깊이, 너무 깊으면 훈련 데이터의 정확도는 높은데, 테스트 데이터의 정확도가
# 					낮아지는 과대적합 현상이 발생합니다.
# 		둘다 하이퍼 파라미터 ( 직접 알아내야하는 값 )

# 9. 모델을 훈련시킵니다.

model.fit( x_train, y_train )

# 10. 테스트 데이터를 예측합니다.

result = model.predict( x_test )  
print(result)

# 11. 모델을 평가합니다.

print ( sum( result == y_test ) / len(y_test) )      # 1.0 / 성능 개선 필요 없겠다.

## 이원교차표를 한번 보자

from sklearn.metrics import confusion_matrix

print ( confusion_matrix( y_test, result ) )             # 순서는 ( 실제, 예측 )
   
# [[820   0]
#  [  0 805]]










