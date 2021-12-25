# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 23:25:23 2021

@author: parkk
"""

# 유방암 데이터를 파이썬으로 분류하는 전체 코드를 가져다가 
# 정확도와 , 카파통계량, 민감도, 특이도를 각각 출력하시오 !

#1. 데이터를 로드합니다.
import  pandas  as  pd

wbcd = pd.read_csv("c:\\data\\wisc_bc_data.csv")

# 데이터를 정규화 합니다.
from   sklearn.preprocessing  import  MinMaxScaler

wbcd2 = wbcd.iloc[ :  , 2: ]  # 환자번호와 diagnosis 제외합니다. 

scaler = MinMaxScaler()   
scaler.fit(wbcd2)   #  최대최소법으로 데이터를 계산합니다.
wbcd2_scaled = scaler.transform(wbcd2)  # 위에서 계산한 내용으로 데이터를 

y = wbcd['diagnosis'].to_numpy()    # 정답 데이터를 numpy array 로 변경합니다.

# 훈련데이터와 테스트데이터로 데이터를 분리합니다.(훈련90%, 테스트:10%)

from sklearn.model_selection  import  train_test_split 

x_train, x_test, y_train, y_test = train_test_split( wbcd2_scaled, y, test_size=0.1, random_state=1)

#모델을 설정합니다.
from  sklearn.neighbors   import  KNeighborsClassifier

model = KNeighborsClassifier( n_neighbors= 5 )  # knn 모델생성

#모델을 훈련시킵니다.
model.fit( x_train, y_train )

#훈련된 모델로 테스트 데이터를 예측합니다.
result = model.predict(x_test)

#모델을 평가합니다.

from  sklearn.metrics  import  accuracy_score

acurracy = accuracy_score( y_test, result)
print( acurracy )                                                # 0.9824561403508771

#12. 모델의 성능을 높입니다.

from  sklearn.metrics  import  confusion_matrix
a = confusion_matrix( y_test, result )
print(a)

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix( y_test, result ).ravel()
print(tn)         # 43
print(fp)         # 0
print(fn)         # 1
print(tp)         # 13

# 13. 카파 통계량도 출력해보자

from sklearn.metrics import cohen_kappa_score

print( cohen_kappa_score ( y_test , result  ) )            # 0.9514893617021276

# ※ 결과 : 

# *정확도 :  0.982

# *confusion matrix :

				     예측       실제 
 #            TN(43)          FP(0)
 #   		  FN(1)            TP(13)

# *카파 통계량 : 0.951

# *민감도와 특이도 :

			  #      TP                                    13
# 민감도 = --------------------------  = -----------------------   = 13/14 = 0.9286 
			#    TP + FN                          13 + 1

print( tp / ( tp + fn ) )

                #     TN                                 43
# 특이도 = --------------------------- = ----------------------------- = 1
            #       TN + FP                             43 + 0

print( tn / ( tn + fp ) )


        #   					knn 모델 
# 정확도                           98.2%     
# 카파통계량                       0.951       
# 민감도                           0.9286           
# 특이도                             1          






