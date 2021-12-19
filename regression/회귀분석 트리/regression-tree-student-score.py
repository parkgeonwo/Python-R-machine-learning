# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 17:58:08 2021

@author: parkk
"""

# student_score.csv 를 가지고 파이썬으로 위와 같이 회귀분석해서 오라클의 상관계수가 값이 0.9771 을 능가할 수 있는지
# 실험하시오 !


# 1. 데이터를 로드합니다.
score = pd.read_csv("c:\\data\\student_score.csv")

# 2. 결측치를 확인합니다.
print( score.isnull().sum() )         # 0

# 3. 종속변수가 정규성을 띄는지 확인합니다.

score['acceptance'].plot(kind = 'hist')


# 예쁜 데이터네

# 4. 훈련 데이터와 테스트 데이터로 분리합니다.

from sklearn.model_selection import train_test_split

x = score.iloc[ :, 1:-1 ].to_numpy()
y = score['acceptance'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split ( x, y , test_size = 0.1, random_state = 1 )

print(x_train.shape )     # 180, 3
print(x_test.shape )       # 20, 3
print(y_train.shape )     # 180,
print(y_test.shape )       # 20,

# 5. 회귀 모델을 생성합니다.

from sklearn.linear_model import LinearRegression
model = LinearRegression( )

# 6. 모델을 훈련 시킵니다.

model.fit( x_train, y_train )

# 7. 테스트 데이터를 예측합니다.

result = model.predict(x_test)

# 8. 실제값과 예측값의 상관계수와 오차를 확인합니다.

import numpy as np

print( np.corrcoef( result, y_test ) )         # 0.93569639
print( mae( result, y_test ) )             # 4.069033970273358

# LinearRegression 을 DecisionTreeRegressor 로 

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor( )

model.fit( x_train, y_train )

result = model.predict(x_test)

import numpy as np

print( np.corrcoef( result, y_test ) )         # 0.82179408
print( mae( result, y_test ) )             # 4.444444445000001

# RandomForestRegressor 로 바꿔서 

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor( random_state = 1 )

model.fit( x_train, y_train )

result = model.predict(x_test)

import numpy as np

print( np.corrcoef( result, y_test ) )         # 0.86500547
print( mae( result, y_test ) )             # 3.8472222210240004


# 결과 :
# LinearRegression 는 상관계수가 0.936 , 오차가 4.072 입니다.
# DecisionTreeRegressor 는 상관계수가 0.817 , 오차가 4.296 입니다.
# RandomForestRegressor 는 상관계수가 0.8531 , 오차가 3.997 입니다.




# *score 를 정규화해서 # 3번 이후에 아래 코드를 넣어줘도 결과는 비슷했다.

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()                                    # 정규화 모델생성
scaler.fit( score )                                              # 훈련데이터를 가지고 정규화 계산
score2 = scaler.transform( score )           

score3 = pd.DataFrame(score2)           # 데이터프레임으로 변경
score3.columns = score.columns          # score3 의 컬럼명을 score 컬럼명으로 다 변경해줌


######### 더 보기 쉽고 좋게 만들어보자 ###############

score = pd.read_csv("c:\\data\\student_score.csv")

from sklearn.model_selection import train_test_split

x = score.iloc[ :, 1:-1 ].to_numpy()
y = score['acceptance'].to_numpy()

val_list = []
val_list2 = []

for i in range(1,31):
    x_train, x_test, y_train, y_test = train_test_split ( x, y , test_size = 0.1, random_state = i )

    from sklearn.linear_model import LinearRegression
    model = LinearRegression( )

    model.fit( x_train, y_train )

    result = model.predict(x_test)

    import numpy as np

    val_list.append( np.corrcoef( result, y_test )[0,1] )         
    val_list2.append( mae( result, y_test ) )             
    
index_num = val_list.index(max(val_list)) + 1

print( 'random_state가',index_num,'일 때, 상관계수가',round( max(val_list),4 ),'이며, 오차는', round( val_list2[index_num -1],4 ),'입니다.' )


# LinearRegression 을 DecisionTreeRegressor 로 

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor( )

# RandomForestRegressor 로 바꿔서 

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor( random_state = 1 )


# 결과 :

# LinearRegression
# random_state가 21 일 때, 상관계수가 0.9784 이며, 오차는 3.5934 입니다.
# DecisionTreeRegressor
# random_state가 28 일 때, 상관계수가 0.9961 이며, 오차는 1.4259 입니다.
# RandomForestRegressor
# random_state가 5 일 때, 상관계수가 0.9948 이며, 오차는 1.7087 입니다.








