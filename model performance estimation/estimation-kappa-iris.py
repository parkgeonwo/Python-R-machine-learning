# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 23:22:27 2021

@author: parkk
"""

# iris 데이터를 분류하는 knn 모델의 정확도와 카파통계량을 같이 출력하시오 ! ( knn 모델은 카페서 끌올 )

# 데이터 로드
import pandas as pd
iris = pd.read_csv("c:\\data\\iris2.csv")        # 결측치, 이상치, 명목형데이터 없음

# 데이터 정규화
from   sklearn.preprocessing  import  MinMaxScaler 
scaler = MinMaxScaler()   
scaler.fit(iris.iloc[:,:-1])
x = scaler.transform(iris.iloc[:,:-1])
y = iris['Species'].to_numpy()

# 훈련/테스트 데이터 분리 (훈련90%, 테스트:10%)
from sklearn.model_selection  import  train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

# my_knn 함수 생성
def my_knn(train, test, label, k):
    pred = []
    for i in range(0,len(test)):
        temp = np.sqrt(np.sum((train - test[i])**2,axis=1))
        result=pd.DataFrame({'rnk': list(map(int,pd.Series(temp).rank())),
                                               'label': label})
        pred.append((result['label'][result.rnk <= k].mode())[0])
    return pred

# 모델 예측
result = my_knn(x_train, x_test, y_train, 1)

# 모델을 평가
from  sklearn.metrics  import  accuracy_score
print ( accuracy_score( y_test, result) )

# 카파 통계량도 출력해보자

from sklearn.metrics import cohen_kappa_score

print( cohen_kappa_score ( y_test , result  ) )            # 0.8863636363636364




