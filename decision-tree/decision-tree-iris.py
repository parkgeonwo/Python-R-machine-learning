# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:07:30 2021

@author: parkk
"""

# 1. 데이터를 로드합니다.
import pandas as pd
iris = pd.read_csv("c:\\data\\iris2.csv")

# 2. 결측치를 확인합니다.
print(iris.isnull().sum())

# 3. 이상치를 확인합니다.

def outlier_value(x):

    for i in list(x.describe ().columns):             # x.columns[x.dtypes =='float64']
        Q1 = x[i].quantile(0.25)
        Q3 = x[i].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + (IQR*1.5)
        lower_bound = Q1 - (IQR*1.5)
        a = x.loc[ (x[i] > upper_bound ) | ( x[i]< lower_bound ) , i ].count()
        b = i
        print( '{0:<10} : {1:>5} 건'.format(  b , a  ) )
        print(x.loc[ (x[i] > upper_bound ) | ( x[i]< lower_bound ) , i ])     # 이상치 행 / 이상치 값을 출력해줌

outlier_value(iris)

# 결과 :
# Sepal.Length :     0 건
# Sepal.Width :     4 건
# 15    4.4
# 32    4.1
# 33    4.2
# 60    2.0
# Petal.Length :     0 건
# Petal.Width :     0 건

# Sepal.Width 이상치 4개를 Sepal.Width의 평균값으로 치환해보자

mean = iris[ 'Sepal.Width' ].mean()              # Sepal.Width 컬럼의 평균값을 mean에 담고

iris.loc [ iris['Sepal.Width'].isin([4.4, 4.1, 4.2, 2.0]), 'Sepal.Width' ] = mean       # 이상치를 평균값으로 치환

outlier_value(iris)         # 다시 확인

# 결과 :
# Sepal.Length :     0 건
# Sepal.Width :     0 건
# Petal.Length :     0 건
# Petal.Width :     0 건


# 4. 명목형 데이터를 숫자로 변경합니다.

print( iris.info() )
# 라벨 데이터 빼고 모두 숫자형 입니다.

# 5. 훈련 데이터와 테스트 데이터를 나눕니다.

x = iris.iloc [ :, :-1].to_numpy()         # 정답 컬럼을 제외한 데이터를 numpy array로 변경
y = iris['Species'].to_numpy()           # 라벨데이터 생성 / numpy array로 바꿔줌

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split ( x, y , test_size = 0.2, random_state = 1 )  # x,y 를 numpy array 형태로 받는다.

print(x_train.shape)        # (120, 4)      / 훈련 데이터
print(x_test.shape)          # (30, 4)     / 테스트 데이터
print(y_train.shape)         # (120,)           / 훈련데이터의 라벨
print(y_test.shape)        # (30,)             / 테스트 데이터의 라벨

# 6. 훈련 데이터를 정규화 합니다.

from sklearn.preprocessing import MinMaxScaler           # standardscaler 보다 minmax가 더 잘나오더라 보통

scaler = MinMaxScaler()                                    # 정규화 모델생성
scaler.fit( x_train )                                              # 훈련데이터를 가지고 정규화 계산
x_train2 = scaler.transform( x_train )           # 계산된 내용으로 데이터를 변환해서 x_train2에 담는다.

# 7. 테스트 데이터를 정규화 합니다.

scaler = MinMaxScaler()                                    # 정규화 모델생성
scaler.fit( x_test )                                              # 훈련데이터를 가지고 정규화 계산
x_test2 = scaler.transform( x_test )           # 계산된 내용으로 데이터를 변환해서 x_test2에 담는다.

# 8. 모델 생성

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier ( criterion = 'entropy', max_depth = 4 )

# 9. 모델 훈련

model.fit( x_train2, y_train )

# 10. 모델 예측

result = model.predict( x_test2 )  

# 11. 모델 평가

print ( sum( result == y_test ) / len(y_test) )      #  0.9333333333333333

from sklearn.metrics import confusion_matrix
print ( confusion_matrix( y_test, result ) )             # 순서는 ( 실제, 예측 )

# [[11  0  0]
#  [ 0 11  2]
#  [ 0  0  6]]





