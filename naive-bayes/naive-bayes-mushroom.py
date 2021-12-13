
# 1. 데이터를 로드합니다.

import pandas as pd
mush = pd.read_csv("c:\\data\\mushrooms.csv")

# 2. 결측치를 확인합니다.

print ( mush.isnull().sum() )

# 3. 이상치를 확인합니다.

# 모두 명목형이므로 이상치 확인 불가

# 4. 명목형 데이터가 있는지 확인합니다.

# R 에서는 데이터를 명목형인 상태 그대로 훈련을 시켰는데 파이썬에서는 전부 숫자로 변경해줘야 합니다.

# 정답을 뺀 데이터만 선별합니다.

x = mush.iloc[ : , 1: ]
print( x.head() )

# 정답 데이터를 y 변수에 담습니다.

y = mush.iloc [  : , 0]
print ( y.head() )

# 명목형 데이터를 숫자로 변경합니다.

mush2 = pd.get_dummies(x)
print( mush2.head() )
print( mush2.shape )            # (8124, 117) , 컬럼의 갯수가 23개에서 117개로 늘어났습니다.
print( mush2.info() )             # 전부 숫자인지 확인합니다.

# 5. 데이터를 정규화 합니다.

print ( mush2.describe() )

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(mush2)
mush2_scaled = scaler.transform(mush2)
print( mush2_scaled )       # numpy array 형태로 생성함

y = y.to_numpy()      # 위에서 만든 정답 데이터를 numpy 형태로 만들어줌
print( y )


# 6. 훈련 데이터와 테스트 데이터를 분리합니다.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split ( mush2_scaled, y , test_size = 0.2, random_state = 1 )

print(x_train.shape)         #  (6499, 117)
print(x_test.shape)               #  (1625, 117)


# 7. 나이브 베이즈 모델을 생성합니다.

#	1. BernoulliNB : 이산형 데이터를 분류할 때 적합
#	2. GaussianNB : 연속형 데이터를 분류할 때 적합
#	3. MultinomialNB : 이산형 데이터를 분류할 때 적합

from sklearn.naive_bayes import BernoulliNB
# from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import MultinomialNB

model = BernoulliNB()

# 8. 훈련 데이터와 라벨( 정답 )으로 모델을 훈련시킵니다.

model.fit(x_train, y_train)

# 9. 훈련된 모델로 테스트 데이터를 예측합니다.

result = model.predict(x_test)

# 10. 모델의 성능을 평가합니다.

print ( (sum( result == y_test )) / (len(y_test)) * 100)          # 93.9076923076923

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, result)
print( accuracy )              # 0.939076923076923

from sklearn.metrics import confusion_matrix

a = confusion_matrix( y_test, result )
print(a)
                                    #      식용 독버섯
# [[815   5]        식용          TN        FP                     
#  [ 94 711]]      독버섯      FN        TP
# FN 값이 너무 높다.. 독버섯인데 식용으로 판단하면 죽는사람 많아짐,, ( 베르누이NB 가 성능이 안좋네 )

tn, fp, fn, tp = confusion_matrix( y_test, result ).ravel()
print(tn, fp, fn, tp)


# 11. 모델의 성능을 높입니다.

# from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import MultinomialNB

model2 = GaussianNB( var_smoothing = 0.001 )
model2.fit(x_train, y_train)
result2 = model2.predict(x_test)

print ( (sum( result2 == y_test )) / (len(y_test)) * 100)          # 99.50769230769231

from sklearn.metrics import confusion_matrix
a = confusion_matrix( y_test, result2 )
print(a)

# [[814   6]
#   [  2 803]]


# for loop 문을 이용해서 GaussianNB 모델의 FN 값이 0 이 되는 var_smoothing 값이 무엇인지 알아 내시오 !

for i in range(1, 101):
    # from sklearn.naive_bayes import BernoulliNB
    from sklearn.naive_bayes import GaussianNB
    # from sklearn.naive_bayes import MultinomialNB

    model2 = GaussianNB( var_smoothing = (i /1000 ) ) 
    model2.fit(x_train, y_train)
    result2 = model2.predict(x_test)

    from sklearn.metrics import confusion_matrix
    a = confusion_matrix( y_test, result2 )
    tn, fp, fn, tp = confusion_matrix( y_test, result2 ).ravel()

    if fn == 0:
        print ( (sum( result2 == y_test )) / (len(y_test)) * 100)          # 안나옴,, fn이 2인값이 최선인듯
        print(a)














