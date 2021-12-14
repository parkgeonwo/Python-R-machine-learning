# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:14:35 2021

@author: parkk
"""

# 1. 데이터를 로드
tat = pd.read_csv("c:\\data\\titanic.csv")
print( tat.head() )

# 필요한 컬럼만 선별합니다.
tat2 = tat.iloc[:, 1:10]
print(tat2.info())

# 2. 결측치 확인
print(tat2.isnull().sum())

# 결과 :

# age         177
# embarked      2

# 나이의 결측치를 나이의 평균값으로 치환하세요 ~

mean = tat2['age'].mean()
tat2['age'].fillna(mean , inplace = True)

print(tat2.isnull().sum())

# 결과:
# embarked      2

# embarked 의 결측치를 나이의 평균값으로 치환하세요 ~

mode = tat2['embarked'].mode()[0]
tat2['embarked'].fillna(mode, inplace = True)
print(tat2.isnull().sum())               # 다 없앰


# 3. 이상치 확인

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

outlier_value( tat2 )

# 결과 :

# survived   :     0 건
# pclass     :     0 건
# age        :    66 건
# sibsp      :    46 건
# parch      :   213 건
# fare       :   116 건


# 4. 명목형 데이터 확인

tat3 = pd.get_dummies(tat2)

x = tat3.iloc[:, 1:].to_numpy()
y = tat3.iloc[:,0].to_numpy()

# 5. 훈련 데이터와 테스트 데이터를 분리


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 1)


# 6. 정규화

from sklearn.preprocessing import MinMaxScaler

scaler.fit(x_train)
x_train2 = scaler.transform(x_train)
x_test2 = scaler.transform(x_test)

# 7. 모델 생성

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier( random_state = 1, n_estimators = 100 )

# 8. 모델 훈련

model.fit(x_train2, y_train)

# 9. 모델 예측

result = model.predict(x_test2)

# 10. 모델 평가

print ( sum(result == y_test) / len(y_test) )        #  0.7821229050279329




