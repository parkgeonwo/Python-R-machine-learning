# ■ R로 부스팅 모델 생성하기

# 1. 필요한 패키지를 로드합니다.
# install.packages("adabag")
library(adabag)
set.seed(300)

# 2. 데이터를 로드합니다.
credit <- read.csv("credit.csv", stringsAsFactors = T )

# 3. 훈련 데이터와 테스트 데이터를 분리합니다.
library(caret)
in_train <- createDataPartition( credit$default, p = 0.75 , list = FALSE )

credit_train <- credit[  in_train,  ]
credit_test <- credit[ -in_train,  ]

# 4. 부스팅 모델을 생성합니다.

m_adaboost <- boosting( default~. , data = credit_train )

# 5. 모델 예측

p_adaboost <- predict( m_adaboost, credit_test )

# 6. 모델 평가

sum( p_adaboost$class == credit_test$default ) / length( credit_test$default )       # 0.756

#  부스팅 모델의 bag의 갯수를 100개로 늘려서 정확도를 보시오 !
  
m_adaboost <- boosting( default~. , data = credit_train , nbagg = 100)        # 0.788

# 400개 

m_adaboost <- boosting( default~. , data = credit_train , nbagg = 400)        # 0.756
