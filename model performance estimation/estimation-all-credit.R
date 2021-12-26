# *독일은행데이터 전체 코드

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

set.seed(659)
credit_shuffle <-  credit[ sample( nrow(credit) ),  ]

#5. 데이터를 9 대 1로 나눈다.

train_num <- round( 0.9 * nrow(credit_shuffle), 0) 

credit_train <- credit_shuffle[1:train_num ,  ]
credit_test  <- credit_shuffle[(train_num+1) : nrow(credit_shuffle),  ]

nrow(credit_train)
nrow(credit_test)

#6. C5.0 패키지와 훈련 데이터를 이용해서 모델을 생성한다.

library(C50)
credit_model <- C5.0( credit_train[ ,-17] , credit_train[  , 17], trials=100 )

#7. 위에서 만든 모델을 이용해서 테스트 데이터의 라벨을 예측한다.

credit_result <-  predict( credit_model, credit_test[  , -17] )
credit_result

#8. 이원 교차표로 결과를 확인한다.

library(gmodels)

CrossTable( credit_test[   , 17], credit_result )

credit_test_prob <- predict ( credit_model,  credit_test[  , -17], type="prob" )
credit_test_prob


credit_results <- data.frame( actural_type=credit_test[  , 17],
                              predict_type=credit_result,
                              prob_yes = round(credit_test_prob[ , 2], 5),
                              prob_no = round( credit_test_prob[ , 1], 5)  )

credit_results

# write.csv( credit_results, 'd:\\data\\final_results.csv', row.names=FALSE)

#install.packages("ROCR")
library(ROCR)
head(credit_results)

pred <- prediction( predictions= credit_results$prob_yes,
                    labels = credit_results$actural_type) 

# 정확도와 cutoff 출력하는 부분 
# perf <- performance(pred, measure = "tpr", x.measure = "fpr")
eval <- performance(pred,"acc")  # y 축을 정확도로 출력
eval  # 392개의 데이터 포인트 추출 
plot(eval)

#설명:  h는 수평선, v 가 수직선의 지점 

#Identifying the best cutoff  and Accuracy
eval #  x축이 cutoff 이고 y축이 정확도를 그래프로 시각화 하기 위한 392개의 데이터 포인트

slot(eval,"y.values") # 392개의 데이터 포인트를 출력 
max <- which.max(slot(eval,"y.values")[[1]])   # 392개의 데이터 포인트중에 max 값에 해당하는
# 인덱스 번호 출력 
max  # 19번째 인덱스 번호에 해당하는 데이터가 가장 큰 값

acc <- slot(eval,"y.values")[[1]][[max]]  #   y축에 정확도중에 61번째에 해당하는 값을 출력
cut <- slot(eval,"x.values")[[1]][[max]]  #   x축에 cutoff 값들중에서 61번째 해당하는 값을 출력
print(c(Accuracy=acc, Cutoff = cut))


perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)
max
tpr <- slot(perf,"y.values")[[1]][[max]]
fpr <- slot(perf,"x.values")[[1]][[max]]
print(c(tpr,fpr))
abline(h= 0.62500000, v=0.03947368)


#■다른 성능척도 총정리 코드:

#■ 실제값과 예측값 대입

credit_test_prob <- predict(credit_model, credit_test[   , -17], type = "prob")
credit_test_prob

# combine the results into a data frame
credit_results <- data.frame(actual_type =credit_test[  , 17],
                             predict_type = credit_result,
                             prob_yes = round(credit_test_prob[ , 2], 5),
                             prob_no = round(credit_test_prob[ , 1], 5))

#3. 예측 데이터 프레임을 csv 로 저장합니다.
# uncomment this line to output the sms_results to CSV
# write.csv(credit_results, "final_results.csv", row.names = FALSE)


#■ 실제값과 예측값 대입

actual_type <- credit_test[  , 17]
predict_type <-  credit_result
positive_value <- 'yes'
negative_value <- 'no'

#■ 정확도

g <- CrossTable( actual_type, predict_type )
x <- sum(g$prop.tbl *diag(2))   # 정확도 확인하는 코드
x

#■ 카파통계량 
#install.packages("vcd")
library(vcd)
Kappa( table( actual_type, predict_type)  ) 

#■ 민감도
#install.packages("caret")
library(caret)
sensitivity( predict_type, actual_type,  positive=positive_value)

#■ 특이도
specificity(  predict_type, actual_type, negative=negative_value)  

#■ 정밀도
posPredValue( predict_type, actual_type, positive=positive_value) 

#■ 재현율 
sensitivity( predict_type, actual_type,  positive=positive_value) 


# calculate AUC
perf.auc <- performance(pred, measure = "auc")
str(perf.auc)
unlist(perf.auc@y.values)


#■ F척도

#1. F1 score 공식
# Fmeasure <- 2 * precision * recall / (precision + recall)

#2. 패키지를 이용하는 방법 
#install.packages("MLmetrics")
library(MLmetrics)

F1_Score(actual_type, predict_type, positive = positive_value)

