# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:10:00 2021

@author: parkk
"""

# 1. 데이터를 로드

import pandas as pd
credit = pd.read_csv("c:\\data\\credit.csv")
print (credit.head() )
print( credit.shape )          # 1000,17

# 2. 데이터 탐색 ( 결측치 확인 )

print(credit.isnull().sum())

# 3. 데이터 탐색 ( 이상치 확인 )          ★

print(credit.info())


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
        # print(x.loc[ (x[i] > upper_bound ) | ( x[i]< lower_bound ) , i ])     # 이상치 행 / 이상치 값을 출력해줌

outlier_value(credit)

# 결과:
# months_loan_duration :    70 건
# amount     :    72 건
# percent_of_income :     0 건
# years_at_residence :     0 건
# age        :    23 건
# existing_loans_count :     6 건
# dependents :   155 건

# 이상치가 보이는 컬럼들은 일단 두고 모델 생서후에 조정해보기로 함


# 4. 데이터 탐색 ( 명목형 데이터 )

print( credit.info() )

 0   checking_balance      1000 non-null   object
 2   credit_history        1000 non-null   object
 3   purpose               1000 non-null   object
 5   savings_balance       1000 non-null   object
 6   employment_duration   1000 non-null   object
 10  other_credit          1000 non-null   object
 11  housing               1000 non-null   object
 13  job                   1000 non-null   object
 15  phone                 1000 non-null   object
 16  default               1000 non-null   object            # 정답 컬럼

credit2 = pd.get_dummies( credit.iloc[: , :-1] )
print( credit2.info() )

# 5. 훈련 데이터와 테스트 데이터를 분리

x = credit2.to_numpy()                         # 학습시킬 데이터 생성
y = credit.iloc[ : , -1 ].to_numpy()        # 정답 데이터 생성

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split( x,y, test_size = 0.1, random_state = 1 )

print(x_train.shape)         # (900,44)
print(x_test.shape)          # (100,44)
print(y_train.shape)        # (900,)
print(y_test.shape)         # (100,)

# 6. 훈련 데이터로 정규화 계산

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)              # 훈련데이터 min/max 정규화 계산


# 7. 계산된 내용으로 훈련데이터를 변형       ★

x_train2 = scaler.transform(x_train)


# 7-2. 계산된 내용으로 테스트 데이터를 변형

x_test2 = scaler.transform(x_test)

print( x_train2.max() , x_train2.min() )           # 1.0 , 0.0
print( x_test2.max(), x_test2.min() )             # 1.2142857142857142  ,  0.0  / 훈련데이터로 훈련했더니 1보다 큰값 나옴

# 8. 모델 생성

from sklearn.tree import DecisionTreeClassifier
model =  DecisionTreeClassifier( criterion = 'entropy', max_depth = 5 )

# 9. 모델 훈련

# 모델명.fit( 훈련데이터, 정답 데이터 )

model.fit(x_train2, y_train)

# 10. 모델 예측

result = model.predict( x_test2 )
result

# 11. 모델 평가

print ( sum( result == y_test ) / len(y_test) )     # 0.76

from sklearn.metrics import confusion_matrix
a = confusion_matrix( y_test, result )
print(a)

# [[58 12]
#  [12 18]]

# 12. 모델 개선

# 	1. 의사결정트리 나무의 가지수인 max_depth 를 늘리는 방법
# 	2. 의사결정트리 + 앙상블 기법 = Random forest 로 모델을 변경합니다.
# 	3. 이상치를 보이는 컬럼의 데이터를 평균값으로 변경
# 	4. 도메인 지식이 있다면 파생변수를 생성

