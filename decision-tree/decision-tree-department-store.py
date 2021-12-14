# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:11:41 2021

@author: parkk
"""

# 1. 데이터를 로드

df_x = pd.read_csv("c:\\data\\X_train.csv", encoding = 'euckr')
df_y = pd.read_csv("c:\\data\\y_train.csv")

# ※ 설명 : 성별 0 = 여자, 1 = 남자

# 2. 결측치를 확인

print ( df_x.isnull().sum() )

# ※ 설명 : 환불금액 컬럼이 3500건 중에 2295 건이 결측치 이므로 컬럼을 삭제합니다.

df_x.drop(['환불금액'],axis =1, inplace = True)
print ( df_x.isnull().sum() )

# 3. 이상치 확인

print(df_x.info())

def outlier_value(x):

    for i in list(x.describe ().columns):             # x.columns[x.dtypes =='float64']
        Q1 = x[i].quantile(0.25)
        Q3 = x[i].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + (IQR*3)
        lower_bound = Q1 - (IQR*3)
        a = x.loc[ (x[i] > upper_bound ) | ( x[i]< lower_bound ) , i ].count()
        b = i
        print( '{0:<10} : {1:>5} 건'.format(  b , a  ) )
        # print(x.loc[ (x[i] > upper_bound ) | ( x[i]< lower_bound ) , i ])     # 이상치 행 / 이상치 값을 출력해줌

outlier_value( df_x )

# 결과 : ( 이상치가 많아서 IQR*3으로 실행했다. 그럼에도 많다.. )
# cust_id    :     0 건
# 총구매액       :   164 건
# 최대구매액      :   149 건
# 내점일수       :    92 건
# 내점당구매건수    :    69 건
# 주말방문비율     :     0 건
# 구매주기       :    57 건

# 위에서 보이는 이상치값을 모델 평가할 때 조정해보겠습니다.

# 4. 명목형 변수 확인

print(df_x.info())

df_x2 = pd.get_dummies(df_x)
print(df_x2.head())

# 5. 훈련 데이터와 테스트 데이터 분리

x = df_x2.iloc[:, 1:].to_numpy()
y = df_y['gender'].to_numpy()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 1)

print(x_train.shape)       # ( 2800,72 )
print(x_test.shape)         # ( 700, 72 )
print(y_train.shape)        # ( 2800, )
print(y_test.shape)         # ( 700, )

# 6. 정규화

from sklearn.preprocessing import MinMaxScaler

scaler.fit(x_train)
x_train2 = scaler.transform(x_train)
x_test2 = scaler.transform(x_test)

# 7. 모델생성

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)

# 8. 모델훈련

model.fit(x_train2, y_train)

# 9. 모델예측

result = model.predict(x_test2)

# 10. 모델 평가

print ( sum(result == y_test) / len(y_test) )        #  0.6257142857142857









