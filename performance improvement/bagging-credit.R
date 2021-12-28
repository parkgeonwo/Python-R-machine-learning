# R 코드를 이용해서 배깅 구현하기

# 1. 필요한 패키지 로드
# install.packages("ipred")
library(ipred)
set.seed(300)

# 2. 데이터로드
credit <- read.csv("credit.csv", stringsAsFactors = T)

# 3. 훈련 데이터와 테스트 데이터 분리
library(caret)
in_train <- createDataPartition( credit$default , p = 0.75 , list = FALSE )
credit_train <- credit [ in_train ,  ]       # 훈련 데이터 구성
credit_test <- credit[ -in_train,  ]         # 테스트 데이터 구성

# 4. 배깅 모델 생성
mybag <- bagging( default~. , data = credit_train, nbagg = 25 )
# 설명 : nbagg = 25 은 앙상블에 사용되는 bag 의 갯수를 25개

# 5. 모델 예측
credit_pred <- predict( mybag, credit_test )

# 6. 모델 성능평가
table( credit_pred, credit_test$default )
prop.table( table( credit_pred, credit_test$default ) )

sum( credit_pred == credit_test$default ) / length( credit_test$default )     # 0.752

#  위의 배깅 모델의 bag 의 갯수를 400개로 늘리고 정확도를 확인하시오 !
  
# 4. 배깅 모델 생성
mybag <- bagging( default~. , data = credit_train, nbagg = 400 )      # 0.776



