# ■ 서포트 벡터 머신 실습 1 ( iris 데이터 )

# 1. 데이터 로드
iris <- read.csv("iris2.csv", stringsAsFactors = T)

# 2. 훈련/테스트 데이터 분리
library(caret)
set.seed(1)

in_train <- createDataPartition( iris$Species, p = 0.8, list = F )
train_data <- iris[ in_train, ] 
test_data <- iris[ -in_train, ]

# 3. 모델생성
#install.packages("e1071")
library(e1071)
set.seed(1)

svm_model <- svm( Species~. , data = train_data , kernel = "linear" )

# 4. 모델 예측

result <- predict( svm_model, test_data )

# 5. 모델 평가

sum( result == test_data$Species ) / length( test_data$Species )                 # 0.9666667

# 6. 모델 성능 개선

# 커널 변경 : kernel = "polynomial"

svm_model <- svm( Species~. , data = train_data , kernel = "linear" )
#↓
svm_model <- svm( Species~. , data = train_data , kernel = "polynomial" )       # 0.966667

# ※ 커널을 linear 로 하나 polynomial 로 하나 똑같이 0.966667 정확도가 출력되고 있습니다.

# 커널을 sigmoid 로 변경하고 모델을 생성해서 정확도를 출력하세요 !
  
svm_model <- svm( Species~. , data = train_data , kernel = "sigmoid" )   # 0.866667








