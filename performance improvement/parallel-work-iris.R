######################## 병렬처리 x #############################

# 1. 데이터 로드
iris <- read.csv("iris2.csv", stringsAsFactors = T)

# 2. 훈련 데이터와 테스트 데이터 분리
library(caret)
set.seed(1)
in_train <- createDataPartition( iris$Species, p = 0.8, list = FALSE )

iris_train <- iris[ in_train,  ]
iris_test <- iris[ -in_train,  ]

# 3. rf + 자동화 모델 생성
library(caret)
set.seed(1)
trCtl <- trainControl( method = 'cv', number = 5 )

a <- system.time({
  m <- train( Species~. , data = iris_train, trControl = trCtl, method = "rf" )
})
a

# 4. 모델 예측
result <- predict( m, iris_test )

# 5. 모델 평가
sum( iris_test$Species == result ) / length( iris_test$Species )        # 0.966667

############################# 병렬처리 했을때 ##########################

# 0. 병렬처리를 위한 패키지 library
library(parallel)
library(doParallel)

# 1. 데이터 로드
iris <- read.csv("iris2.csv", stringsAsFactors = T)

# 2. 훈련 데이터와 테스트 데이터 분리
library(caret)
set.seed(1)
in_train <- createDataPartition( iris$Species, p = 0.8, list = FALSE )

iris_train <- iris[ in_train,  ]
iris_test <- iris[ -in_train,  ]

# 3. rf + 자동화 모델 생성
library(caret)
set.seed(1)

trCtrl <- trainControl( method = 'cv', number = 5 )

cl <- makeCluster( detectCores() - 1 )    # 현재 시스템의 코어의 갯수를 마이너스 1개를 cl 넣는다.
cl        # "호스트 ‘localhost’에 있는 3개의 노드들을 가진 소켓클러스터입니다"

registerDoParallel(cl)

a <- system.time({
  m2 <- train( Species~. , data = iris_train, trControl = trCtrl, method = "rf", allowParallel = TRUE )
})
a

# 4. 모델 예측
result2 <- predict( m2, iris_test )

# 5. 모델 평가
sum( iris_test$Species == result2 ) / length( iris_test$Species )        # 0.966667

# 6. 병렬 프로세서 정리하는 방법
stopCluster(cl)

on.exit( stopCluster( cl ) )


# 병렬 x 
# 사용자  시스템 elapsed 
#   1.35    0.13    1.85 

# 병렬 o
# 사용자  시스템 elapsed 
#   0.80    0.03    8.41 
