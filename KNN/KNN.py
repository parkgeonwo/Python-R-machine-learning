# ■ 판다스로 유방암 판정 KNN 모델 생성하기


# Pandas )

# 1. 데이터 로드합니다.

import pandas as pd
wbcd = pd.read_csv("c:\\data\\wisc_bc_data.csv")

# 판다스는 R 과 다르게 stringsAsFactors = TRUE 를 지정하지 않아도 됩니다.

# 2. 데이터를 확인합니다.

wbcd.info()            # 컬럼명과 데이터 타입을 확인합니다.
print( wbcd.shape )         # 몇행 몇열 인가? , (569, 32)
print(wbcd.describe())            # R 에서의 summary() 와 같은 함수

# 3. 결측치를 확인합니다.

print( wbcd.isnull().sum() )

# 4. 이상치를 확인합니다.

def outlier_value(x):

    for i in list(x.describe ().columns):                                # x.columns[x.dtypes =='float64']
        Q1 = x[i].quantile(0.25)
        Q3 = x[i].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + (IQR*5)
        lower_bound = Q1 - (IQR*5)
        a = x.loc[ (x[i] > upper_bound ) | ( x[i]< lower_bound ) , i ].count()
        b = i
        print( '{0:<10} : {1:>5} 건'.format(  b , a  ) )
        
outlier_value(wbcd)

# 설명 : area_se 와 dimension_se 의 이상치가 보이므로 모델평가후에 정확도를 더 높이기 위해서
#             이 두 컬럼의 이상치를 중앙값으로 치환해볼 필요가 있습니다.


# 5. 명목형 데이터가 있는지 확인합니다.

wbcd.info()             # label 만 object 나머지는 수치형

# 6. 데이터를 정규화합니다.

from sklearn.preprocessing import MinMaxScaler

wbcd2 = wbcd.iloc[     :    ,  2:    ]        # 환자번호와 diagnosis를 컬럼을 제외 , 행열번호로 할땐 iloc 사용
# print(wbcd2) 

scaler = MinMaxScaler()

scaler.fit(wbcd2)                   # 최대 최소법으로 데이터를 계산합니다.

wbcd2_scaled = scaler.transform( wbcd2 )            # 위에서 계산한 내용으로 데이터를 변환해서 wbcd2_scaled 담습니다.
# print(wbcd2_scaled)

# print ( wbcd2_scaled.shape )         # (569, 30)   , numpy array 형태로 변경되었습니다.

y = wbcd['diagnosis'].to_numpy()        # 정답 데이터를 numpy array 로 변경합니다.
# print(y)



# 7. 훈련데이터와 테스트데이터로 분리합니다. ( 훈련 90% , 테스트 10% )

from sklearn.model_selection import train_test_split

x_train , x_test, y_train, y_test = train_test_split( wbcd2_scaled, y, test_size = 0.1 , random_state = 1 )    

# 자동으로 shuffle 됨,  test_size = 0.1 은 테스트 10% 한다는것 
# random_state = 1 은 seed =1 ,  seed 값을 정해주는 이유 : 어느 자리에서든 동일한 정확도를 보이는 모델을 만들기 위해서
# x_train : 훈련데이터  , x_test : 테스트 데이터 
# y_train : 훈련데이터의 정답 , y_test : 테스트 데이터의 정답

# print( x_train.shape )      # (512, 30)
# print( x_test.shape )         # (57, 30)
# print( y_train.shape )         # (512,)
# print( y_test.shape )          # (57,)

# 8. 모델을 설정합니다.

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier( n_neighbors = 5  )                 # knn 모델생성, k = 5 넣어준 모델

# 9. 모델을 훈련시킵니다.

model.fit(x_train, y_train)                  # 훈련

# 10. 훈련된 모델로 테스트 데이터를 예측합니다.

result = model.predict( x_test )
# print(result)

# 11. 모델을 평가합니다.

print ( sum( y_test == result ) / 57 *100 )           # 98.24561403508771

또는

from sklearn.metrics import accuracy_score

acurracy = accuracy_score( y_test , result )          # 실제값, 예측값 넣으면 정확도가 나옴
print ( acurracy )                                                # 0.9824561403508771

# 12. 모델의 성능을 높입니다.

from sklearn.metrics import confusion_matrix

a = confusion_matrix( y_test, result )
print(a)

# [[43  0]
#  [ 1 13]]

# 뭐가 TN, FP, FN, TP 인지 알아보자

tn, fp, fn, tp = confusion_matrix( y_test, result ).ravel()

print( tn, fp, fn, tp )          # 43 0 1 13

# [[43  0]                   TN   FP
#  [ 1 13]]                  FN   TP



#%%


# FN 를 0 으로 만들면서 정확도가 가장 좋은 K 값을 무엇인지 알아내세요 ~

for i in range(1,10):
    for k in range(1,51,2):

        import pandas as pd
        wbcd = pd.read_csv("c:\\data\\wisc_bc_data.csv")

        from sklearn.preprocessing import MinMaxScaler
 
        wbcd2 = wbcd.iloc[     :    ,  2:    ]        # 환자번호와 diagnosis를 컬럼을 제외 , 행열번호로 할땐 iloc 사용
        # print(wbcd2) 

        scaler = MinMaxScaler()

        scaler.fit(wbcd2)                   # 최대 최소법으로 데이터를 계산합니다.

        wbcd2_scaled = scaler.transform( wbcd2 )            # 위에서 계산한 내용으로 데이터를 변환해서 wbcd2_scaled 담습니다.
        # print(wbcd2_scaled)

        # print ( wbcd2_scaled.shape )         # (569, 30)   , numpy array 형태로 변경되었습니다.

        y = wbcd['diagnosis'].to_numpy()        # 정답 데이터를 numpy array 로 변경합니다.
        # print(y)

        from sklearn.model_selection import train_test_split

        x_train , x_test, y_train, y_test = train_test_split( wbcd2_scaled, y, test_size = 0.1 , random_state = i )  

        from sklearn.neighbors import KNeighborsClassifier

        model = KNeighborsClassifier( n_neighbors = k  )                 # knn 모델생성, k = i 넣어준 모델

        model.fit(x_train, y_train)                  # 훈련
        result = model.predict( x_test )

        from sklearn.metrics import accuracy_score
  
        acurracy = accuracy_score( y_test , result )          # 실제값, 예측값 넣으면 정확도가 나옴                                            

        from sklearn.metrics import confusion_matrix

        tn, fp, fn, tp = confusion_matrix( y_test, result ).ravel()

        if fn == 0 :
            print('random_state =', k, ', k의 값 =', i)
            print ( acurracy )                                                
            print( tn, fp, fn, tp )         


#%%

# 와인데이터 wine.csv 를 가지고 와인의 종류를 분류하는 머신러닝 모델을 파이썬으로 구현하시오 !

import pandas as pd
wine = pd.read_csv("c:\\data\\wine.csv")

from sklearn.preprocessing import MinMaxScaler
 
wine2 = wine.iloc[     :    ,  1:    ]        # Type 컬럼을 제외 , 행열번호로 할땐 iloc 사용

scaler = MinMaxScaler()

scaler.fit(wine2)                   # 최대 최소법으로 데이터를 계산합니다.

wine2_scaled = scaler.transform( wine2 )    # 위에서 계산한 내용으로 데이터를 변환해서 wine2_scaled 담습니다.

y = wine['Type'].to_numpy()        # 정답 데이터를 numpy array 로 변경합니다.

from sklearn.model_selection import train_test_split

x_train , x_test, y_train, y_test = train_test_split( wine2_scaled, y, test_size = 0.1 , random_state = 1 )  

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier( n_neighbors = 3  )                 # knn 모델생성, k = 넣어준 모델

model.fit(x_train, y_train)                  # 훈련
result = model.predict( x_test )

from sklearn.metrics import accuracy_score

acurracy = accuracy_score( y_test , result )          # 실제값, 예측값 넣으면 정확도가 나옴                                            

from sklearn.metrics import confusion_matrix

a = confusion_matrix( y_test, result )
print(a)
print( acurracy )
