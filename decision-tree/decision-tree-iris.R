#1. 의사결정트리에 필요한 패키지를 설치합니다.
install.packages("C50")

#2. 데이터를 로드합니다.
iris <- read.csv("iris2.csv", stringsAsFactor = T)

#3. 훈련 데이터와 테스트 데이터로 데이터를 분리합니다. ( 훈련 8, 테스트 2 )
library(caret)
set.seed(1)
train_num <- createDataPartition(iris$Species, p = 0.8, list = F)

train_num
length(train_num)          # 120

train_data <- iris[ train_num, ]
test_data <- iris[ -train_num, ]

nrow(train_data)         # 120
nrow(test_data)           # 30

#4. 훈련데이터로 의사결정트리 모델을 만듭니다.

library(C50)
model <- C5.0( train_data[ , -5 ], train_data[ ,5 ] )
#     ↑                                    ↑
# 라벨 제외 훈련데이터    훈련 데이터의 라벨(정답) 

model

#5. 훈련한 모델로 테스트 데이터를 예측합니다.

result <- predict(model, test_data[  , -5 ]   )      

#6. 모델의 성능(정확도)을 확인합니다.

sum( result == test_data[  , 5 ] ) / length(result)  # [1] 0.9333333  = 28/30

#7. 모델의 성능을 높입니다.

model2 <- C5.0( train_data[ , -5 ], train_data[ ,5 ] , trials = 10)
#     ↑                                    ↑
# 라벨 제외 훈련데이터    훈련 데이터의 라벨(정답) 

result2 <- predict(model, test_data[  , -5 ]  )         

sum( result2 == test_data[ , 5] )/length(result2)            # [1] 0.9333333

# 8. trials 를 하지 않은 의사결정트리 모델을 시각화 하시오 !

library(C50)

model <- C5.0( train_data[ , -5 ], train_data[ ,5 ] )
plot(model)
