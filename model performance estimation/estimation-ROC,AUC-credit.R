# 독일은행의 채무불이행자를 예측하는 모델의 ROC 커브를 그리시오 !

#1. 데이터를 로드한다.

credit <- read.csv("credit.csv", stringsAsFactor=TRUE)
str(credit) 

#2. 데이터에 각 컬럼들을 이해한다. 

#라벨 컬럼 :  default  --->  yes : 대출금 상환 안함 
#no  : 대출금 상환 

prop.table( table(credit$default)  )
summary( credit$amount)

#3. 데이터가 명목형 데이터인지 확인해본다.

str(credit) 

#4. 데이터를 shuffle 시킨다.

set.seed(31)
credit_shuffle <-  credit[ sample( nrow(credit) ),  ]

#5. 데이터를 9 대 1로 나눈다.

train_num <- round( 0.9 * nrow(credit_shuffle), 0) 

credit_train <- credit_shuffle[1:train_num ,  ]

credit_test  <- credit_shuffle[(train_num+1) : nrow(credit_shuffle),  ]


#6. C5.0 패키지와 훈련 데이터를 이용해서 모델을 생성한다.

library(C50)

credit_model <- C5.0( credit_train[ ,-17] , credit_train[  , 17] )

#7. 위에서 만든 모델을 이용해서 테스트 데이터의 라벨을 예측한다.

credit_result <-  predict( credit_model, credit_test[  , -17] )                      # no 가 채무이행, yes 가 채무불이행

#8. 이원 교차표로 결과를 확인한다.

library(gmodels)

CrossTable( credit_test[   , 17], credit_result )

# 설명 : 정확도 77%의 모델입니다.

# 9. 실제값과 예측값 그리고 예측 확률을 담는 credit_results 라는 데이터 프레임을 생성합니다.

credit_test_prob <- predict( credit_model , credit_test[ , -17 ] , type = "prob" )
credit_test_prob          # no 일 확률, yes 일 확률



# 설명 : type = "prob" 옵션을 주게 되면 확률이 출력됩니다.

credit_results <- data.frame( actual_type = credit_test[ , 17 ],
                              predict_type = credit_result,
                              prob_yes = round( credit_test_prob[  ,2 ],5 ),
                              prob_no = round( credit_test_prob[ , 1 ], 5 )
)

credit_results

# 10. credit_results 데이터 프레임을 csv 파일로 저장합니다.

write.csv( credit_results, "c:\\data\\final_results.csv", row.names = FALSE )

# 11. ROC 커브 그래프 그리기

install.packages("ROCR")
library(ROCR)

pred <- prediction( predictions = credit_results$prob_yes,
                    labels = credit_results$actual_type )
pred

# 설명 : prediction( predictions = 관심범주의 확률, labels = 실제정답 )
# pred 에 담기는게 ROC 커브를 그리기 위한 100개의 데이터 포인트가 담긴다.

perf <- performance( pred, measure = 'tpr', x.measure = 'fpr' )
perf

# 설명 : roc 커브의 x 축인 fpr , y축인 tpr 에 해당되는 실제 data point 24개가 생성됨

plot( perf, main = "ROC 커브", col = 'blue', lwd = 2 )



# 대각선 출력

abline( a = 0, b = 1, lwd = 2, lty = 2 )

# 설명 : a는 직선의 절편, 1 은 직선의 기울기, lwd 는 선의 굵기 , lty = 2는 점선으로 표현



# 12. AUC 계산하기

# AUC ? Area Under Curve 의 약자로 곡선 아래쪽의 넓이를 말합니다.

value <- performance( pred, measure = "auc" )
str( value )
unlist( value@y.values )       # 0.6668286  / 1에 가까울수록 좋은것이다.




