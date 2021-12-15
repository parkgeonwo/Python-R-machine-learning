# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 15:41:07 2021

@author: parkk
"""

# 1. 데이터 로드

import pandas as pd
mush = pd.read_csv("c:\\data\\mushrooms.csv")

# 2. 훈련 데이터와 테스트 데이터 분리

x = mush.iloc[ : , 1:  ]
y = mush.iloc[ : , 0 ]

# LabelEncoder 를 사용해서 정답 컬럼을 0 과 1로 변경합니다.

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
y2 = encoder.transform(y)               # type(y2) = numpy array

# 관심범주 확인하는 코드

print(encoder.classes_)                 # ['edible' 'poisonous']   / 0 , 1

x2 = x.to_numpy()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x2, y2, test_size = 0.2 , random_state = 1 )

print(x_train.shape)        # ( 6499,22 )
print(x_test.shape)        # ( 1625,22 )
print(y_train.shape)        # ( 6499, )
print(y_test.shape)        # ( 1625, )

# 3. 모델 생성

import wittgenstein as lw
model = lw.RIPPER()

# 4. 모델 훈련

model.fit(x_train , y_train)

# 5. 모델 예측

result = model.predict(x_test)

# 6. 모델 평가

print(sum(result == y_test) / len(y_test))        # 1 , 다르게 나왔다면 model.fit 에서 random_state 조절









