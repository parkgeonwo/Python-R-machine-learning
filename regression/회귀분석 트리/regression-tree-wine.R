# ▩ 와인 데이터의 등급 ( 수치 ) 를 예측하는 회귀트리 모델을 생성하는 실습 ( p294 )
# ( whitewines.csv )

# 1. 데이터를 로드합니다.

wine <- read.csv('whitewines.csv')
head(wine)

#fixed.acidity       : 고정 산도
#volatile.acidity    : 휘발성 산도
#citric.acid         : 시트르산
#residual.sugar      : 잔류 설탕
#chlorides           : 염화물
#free.sulfur.dioxide : 자유 이산화황
#total.sulfur.dioxide: 총 이산화황
#density             : 밀도
#pH                  : pH
#sulphates           : 황산염
#alcohol             : 알코올
#quality             : 품질 <------------ 종속변수 입니다.



# 2. 종속변수인 quality 가 정규분포에 속하는지 확인합니다.

hist(wine$quality)
# 설명 : 어느 한쪽으로 데이터가 치우치지 않은 안정적인 모양을 보이고 있습니다.




# 3. 결측치가 있는지 확인합니다.

colSums(is.na(wine))

# 4. 훈련 데이터와 테스트 데이터로 데이터를 분리 합니다.

library(caret)
set.seed(1)
train_num <- createDataPartition( wine$quality , p = 0.9, list = F )
train_data <- wine[ train_num,  ]
test_data <- wine[ -train_num,  ]

nrow(train_data)        # 4409
nrow(test_data)          # 489

# 5. 훈련 데이터로 모델을 생성합니다.

install.packages("rpart")
library(rpart)

model <- rpart( quality ~. , data = train_data )
model

# 1) root 4409 3430.35200 5.877977  
# 2) alcohol< 10.85 2769 1642.60700 5.603467  
# 4) volatile.acidity>=0.2525 1449  697.05730 5.363009 *
#   5) volatile.acidity< 0.2525 1320  769.79920 5.867424  
# 10) volatile.acidity>=0.2075 656  325.38870 5.708841 *
#   11) volatile.acidity< 0.2075 664  411.61450 6.024096  
# 22) residual.sugar< 12.65 530  294.09250 5.903774 *
#   23) residual.sugar>=12.65 134   79.50000 6.500000 *
#   3) alcohol>=10.85 1640 1226.78000 6.341463  
# 6) free.sulfur.dioxide< 10.5 89   98.76404 5.370787 *
#   7) free.sulfur.dioxide>=10.5 1551 1039.34800 6.397163  
# 14) alcohol< 11.74167 752  482.56250 6.187500 *
#   15) alcohol>=11.74167 799  492.61580 6.594493 *
  
  
# ※ 설명 : * 표시가 있는 노드는 앞노드로 노드에서 예측이 이루어진다는 것을 의미합니다.
# 와인 데이터의 예측 등급입니다.

# quality 5.9 입노드로 예를 들면 alcohol< 10.85 이고 volatile.acidity>=0.2525 이면서
# volatile.acidity < 0.2075 고  residual.sugar< 12.65 이면 이 와인의 quality 는 5.9로 예상됩니다.
# quality 가 3~9 등급 사이로 구성되어져 있다.

# 6. 생성된 모델을 시각화 합니다.

install.packages("rpart.plot")
library(rpart.plot)

rpart.plot(model, digits = 3)

# ※ 설명 : digits = 3 은 소수점 세번째까지 허용하겠다는 뜻




# 7. 훈련된 모델로 테스트 데이터를 예측합니다.

result <- predict( model , test_data[ , -12 ] )
result

# 8. 예측값과 실제값의 상관계수를 구하여 모델의 성능을 평가합니다.

cor( result , test_data[ , 12 ] )          # 0.51

# 9. 예측값과 실제값의 오차율을 확인하여 모델의 성능을 평가합니다.

mae <- function( actual, predicted ) { mean( abs( actual - predicted ) ) } 
# 실제값에서 예측값을 뺸 절대값들의 평균 = 오차

mae( result, test_data[ , 12 ] )     # 0.64

# 상관계수는 1에 가까워야하고 오차는 0에 가까워야 좋은 모델이다.

# ※ 설명 : 이 모델의 경우 다른 모델인 서포트 벡터 머신에서는 오차가 0.45인데 
# 0.64 이면 상대적으로 큰 오차이므로 개선의 여지가 필요해 보입니다.

# 개선 방법 : 회귀트리 -------------> 모델트리

# ▩ 모델트리 ( p 306 )


# 기존회귀 트리 모델 + 다중회귀를 추가한 모델

# 회귀트리는 무조건 분할한 y ( 종속변수의 값들 ) 값들의 평균값으로만 예측을 했는데
# 모델트리는 분할한 x 값과 y 값들에 대한 회귀식을 통해서 y 값을 예측합니다.

# 1. 데이터를 로드합니다.
wine <- read.csv("whitewines.csv")

# 2. 와인 데이터 종속변수의 분포를 확인합니다.
hist(wine$quality)

# 3. 와인 데이터를 훈련과 테스트로 분리합니다.

library(caret)
set.seed(1)
train_num <- createDataPartition( wine$quality , p = 0.9, list = F )
train_data <- wine[ train_num,  ]
test_data <- wine[ -train_num,  ]

nrow(train_data)        # 4409
nrow(test_data)          # 489

# 4. 모델트리를 구현하기 위한 패키지를 설치

install.packages("Cubist")
library(Cubist)

# 5. 와인의 품질을 예측하는 모델을 생성합니다.

model2 <- cubist( x = train_data[  , -12 ], y = train_data[  , 12 ] )
model2

# 6. 만든 모델로 테스트 데이터를 예측합니다.

result2 <- predict( model2, test_data[  , -12 ] )
result2

# 7. 실제값과 예측값간의 상관계수와 오차를 확인합니다.

cor ( result2 , test_data[  , 12 ] )      # 0.5954519

mae( result2 , test_data[  ,12 ] )    # 0.573

# ※ 설명 : 회귀트리일때는 오차가 0.64 였는데 0.57이면 많이 개선되었습니다.
