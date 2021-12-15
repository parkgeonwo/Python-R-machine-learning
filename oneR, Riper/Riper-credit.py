# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 15:47:34 2021

@author: parkk
"""

# 1. 데이터 로드

import pandas as pd
credit = pd.read_csv("c:\\data\\credit.csv")

# 2. 정답 컬럼 변경 & 명목형 데이터 변경

x = credit.iloc[ : , 0:-1  ]
y = credit.iloc[ : , -1 ]

# LabelEncoder 를 사용해서 정답 컬럼을 0 과 1로 변경합니다.

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
y2 = encoder.transform(y)               # type(y2) = numpy array

# 관심범주 확인하는 코드

print(encoder.classes_)                 # ['no' 'yes']   / 0 , 1

# 명목형 데이터 변경

x2 = pd.get_dummies(x)
x3 = x2.to_numpy()

# 3. 훈련데이터 / 테스트 데이터 분리

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x3, y2, test_size = 0.1 , random_state = 1 )

print(x_train.shape)        # ( 800 ,44 )
print(x_test.shape)        # ( 200, 44 )
print(y_train.shape)        # ( 800, )
print(y_test.shape)        # ( 200, )

# 3. 모델 생성

import wittgenstein as lw
model = lw.RIPPER()

# 4. 모델 훈련

model.fit(x_train , y_train, random_state = 5 )

# 5. 모델 예측

result = model.predict(x_test)

# 6. 모델 평가

print(sum(result == y_test) / len(y_test))        # 0.8





