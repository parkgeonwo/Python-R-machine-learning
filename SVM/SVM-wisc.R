#  유방암 데이터의 악성과 양성을 분류하는 머신러닝 모델을 서포트 벡터 머신으로 생성하고
# 정확도를 확인하시오 ! ( 훈련 8, 테스트 2 )

# 1. 데이터 로드
wisc <- read.csv("wisc_bc_data.csv", stringsAsFactors = T)

# 2. 훈련 데이터와 테스트 데이터 분리
library(caret)
set.seed(1)
in_train <- createDataPartition( wisc$diagnosis, p = 0.8, list = FALSE )

wisc_train <- wisc[ in_train,  ]
wisc_test <- wisc[ -in_train,  ]

# 3. 모델 생성
library(ipred)
library(caret)
library(e1071)
set.seed(1)

m <- train( diagnosis~. , data = wisc_train, method = "svmLinear" )

# 4. 모델 예측
result <- predict( m, wisc_test )

# 5. 모델 평가
sum( wisc_test$diagnosis == result ) / length( wisc_test$diagnosis )        # 0.9823009

# svmRadial 했을땐 정확도 0.9734513
