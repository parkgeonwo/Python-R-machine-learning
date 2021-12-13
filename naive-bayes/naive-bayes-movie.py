# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 21:07:28 2021

@author: parkk
"""



# 나이가 20대이고 성별이 여자이며 직업이 IT 이고 결혼을 안했으며
# 이성친구가 없는 사람이 선택할 가능성이 높은 영화 장르는 ?  ( 파이썬으로, moive.csv )


import pandas as pd
movie = pd.read_csv("c:\\data\\movie.csv", encoding = 'euckr')
# print ( movie.isnull().sum() )     # 결측치확인

x = movie.iloc[ : , :5 ]            # 정답을 뺀 데이터 생성
y = movie.iloc [  : , 5]            # 정답 데이터 생성

movie2 = pd.get_dummies(x)  #  5개 컬럼에서 18개로 늘어남 , 숫자 데이터로 변환

from sklearn.preprocessing import MinMaxScaler         # 데이터를 정규화 합니다.

scaler = MinMaxScaler()
scaler.fit(movie2)
movie2_scaled = scaler.transform(movie2)

y = y.to_numpy()           # y를 numpy로 만들어줌

from sklearn.model_selection import train_test_split       # 훈련데이터와 테스트 데이터 분리

x_train, x_test, y_train, y_test = train_test_split ( movie2_scaled, y , test_size = 0.2, random_state = 1 )

from sklearn.naive_bayes import BernoulliNB      # 나이브 베이즈 모델 생성
 
model = BernoulliNB()

model.fit(x_train, y_train)          # 모델 훈련

result = model.predict(x_test)          # 예측

print ( (sum( result == y_test )) / (len(y_test)) * 100)          # 87.5

# 성능 향상 & var_smoothing 값 찾기

x_val = list(range(1,1000))
y_val = []

for i in range(1, 1000):
    # from sklearn.naive_bayes import BernoulliNB
    from sklearn.naive_bayes import GaussianNB
    # from sklearn.naive_bayes import MultinomialNB

    model2 = GaussianNB( var_smoothing = (i /1000 ) ) 
    model2.fit(x_train, y_train)
    result2 = model2.predict(x_test)

    from sklearn.metrics import confusion_matrix
    a = sum( result2 == y_test ) / (len(y_test)) * 100
    y_val.append(a)    
                              
plot_dict = { 'i' : x_val, '확률' : y_val }
plot_dict2 = pd.DataFrame(plot_dict)
print(plot_dict2.plot(kind = 'line', x = 'i', y = '확률'))


# i = 599 이상일때 ( var_smoothing가 0.599 이상 ) 확률이 100이다 (a == 100) 

######  예측할 데이터 생성

print(movie2.columns)       # 숫자로 변형한 데이터의 컬럼 확인
						
# ['나이_10대', '나이_20대', '나이_30대', '나이_40대', '성별_남', '성별_여', '직업_IT',
# '직업_디자이너', '직업_무직', '직업_언론', '직업_영업', '직업_자영업', '직업_학생', '직업_홍보/마케팅',
 # '결혼여부_NO', '결혼여부_YES', '이성친구_NO', '이성친구_YES']

exp_data = {}
temp_list = [0]*len(movie2.columns)

for i,k in zip(movie2.columns, temp_list ):
    exp_data[i] = k 

exp_data['나이_20대'] = 1
exp_data['직업_IT'] = 1
exp_data['성별_여'] = 1
exp_data['결혼여부_NO'] = 1
exp_data['이성친구_NO'] = 1

exp_data2 = pd.DataFrame ( exp_data, index = [0] )   

# 결과 예측
result2 = model2.predict(exp_data2.to_numpy())          # 예측
print(result2)                   # 정답 : 로맨틱



















