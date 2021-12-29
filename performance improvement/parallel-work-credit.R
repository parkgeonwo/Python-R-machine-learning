# 1. R 에서 코어의 갯수 확인하기
# install.packages("future")         # 병렬처리를 쉽게 구현해주는 패키지
library(future)

availableCores()

# 2. 병렬작업을 위해 추가로 패키지 설치

# install.packages("future.apply")
# install.packages("furrr")

library(future.apply)

# ▩  1. 독일은행 데이터의 채무 불이행자 예측하는 배깅 모델을 병렬 작업 없이 수행

# 1. 데이터로드
credit <- read.csv("credit.csv", stringsAsFactors = T)

# 2. 훈련/테스트 데이터 분리
library(caret)
set.seed(1)
in_train <- createDataPartition( credit$default, p = 0.8, list = FALSE )

credit_train <- credit[ in_train,  ]
credit_test <- credit[ -in_train,  ]

# 3. 앙상블 + 자동튜닝 모델 생성 ( 병렬작업 x )
library(adabag)
library(caret)
trCtl <- trainControl( method = 'cv' , number = 10 )
set.seed(1)

a <- system.time({
  m2 <- train( default~. , data = credit_train , trControl = trCtl, method = 'treebag' )
})
a

# 사용자  시스템 elapsed 
# 8.62    0.18   10.11 

# 4. 모델 예측
result2 <- predict( m2, credit_test )

# 5. 모델 평가
sum( credit_test$default == result2 ) / length( credit_test$default )           # 0.775

# ▩  2. 독일은행 데이터의 채무 불이행자 예측하는 배깅 모델을 병렬으로 수행

# 0. 병렬처리를 위한 패키지 library
library(parallel)
library(doParallel)

# 1. 데이터로드
credit <- read.csv("credit.csv", stringsAsFactors = T)

# 2. 훈련/테스트 데이터 분리
library(caret)
set.seed(1)
in_train <- createDataPartition( credit$default, p = 0.8, list = FALSE )

credit_train <- credit[ in_train,  ]
credit_test <- credit[ -in_train,  ]

# 3. 앙상블 + 자동튜닝 모델 생성 ( 병렬작업 x )
library(adabag)
library(caret)
trCtrl <- trainControl( method = 'cv' , number = 10 )

cl <- makeCluster( detectCores() - 1 )    # 현재 시스템의 코어의 갯수를 마이너스 1개를 cl 넣는다.
cl        # "호스트 ‘localhost’에 있는 3개의 노드들을 가진 소켓클러스터입니다"

set.seed(1)

registerDoParallel(cl)

a <- system.time({
  m2 <- train( default~. , data = credit_train , trControl = trCtrl, method = 'treebag', allowParallel = TRUE )
})
a

# 사용자  시스템 elapsed 
# 1.39    0.05   13.37

# 4. 모델 예측
result2 <- predict( m2, credit_test )

# 5. 모델 평가
sum( credit_test$default == result2 ) / length( credit_test$default )           # 0.775

# 6. 병렬 프로세서 정리하는 방법
stopCluster(cl)

on.exit( stopCluster( cl ) )
