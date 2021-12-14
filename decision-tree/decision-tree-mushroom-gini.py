# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:06:57 2021

@author: parkk
"""

import pandas as pd
mush = pd.read_csv("c:\\data\\mushrooms.csv")

mush2 = pd.get_dummies( mush.iloc [ :, 1: ]   )            # 정답 컬럼제외하고 숫자형으로 변경

x = mush2.to_numpy()                      # 숫자로 변경한 훈련데이터를 numpy array로 변경
y = mush['type'].to_numpy()           # 라벨데이터 생성 / numpy array로 바꿔줌

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split ( x, y , test_size = 0.2, random_state = 1 )  # x,y 를 numpy array 형태로 받는다.

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier ( criterion = 'gini', max_depth = 5 )       #  max_depth : 가지의 깊이
model.fit( x_train, y_train )

result = model.predict( x_test )  

print ( sum( result == y_test ) / len(y_test) )      # 0.9987692307692307

from sklearn.metrics import confusion_matrix
print ( confusion_matrix( y_test, result ) )             # 순서는 ( 실제, 예측 )

# [[820   0]
#  [  2 803]]

# gini 지수 테스트 한 결과 entropy 로 했을 때 보다 정확도가 떨어졌다. fn 값이 2가 증가 했습니다.









