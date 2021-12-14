nrow(skin)           # 30건 밖에 안되므로 29건으로 학습 시키고 1건으로 테스트 합니다.

#1. 의사결정트리에 필요한 패키지를 설치합니다.
#2. 화장품 고객 데이터를 로드합니다.
#3. 화장품 고객 데이터를 훈련 ( 20개 ), 테스트 (10개) 로 나눕니다.
#4. 분류 모델을 만듭니다. 
#5. 훈련한 모델로 테스트 데이터 10개를 예측합니다.
#6. 모델의 성능(정확도)을 확인합니다.
#7. 모델의 성능을 높입니다.


#1. 의사결정트리에 필요한 패키지를 설치합니다.

install.packages("C50")

#2. 화장품 고객 데이터를 로드합니다.

skin <- read.csv("skin.csv", stringsAsFactor = T)
head(skin)

#3. 화장품 고객 데이터를 훈련 ( 20개 ), 테스트 (10개) 로 나눕니다.

library(caret)
set.seed(1)
train_num <- createDataPartition(skin$cupon_react, p = 0.8, list = F)

train_num
length(train_num)          # 25

train_data <- skin[ train_num, ]
test_data <- skin[ -train_num, ]

nrow(train_data)         # 25 ,        컬럼에 문자와 숫자가 섞여있다!! (나이브,knn과 다르네)
nrow(test_data)           # 5

#4. 분류 모델을 만듭니다. 

library(C50)
model <- C5.0( train_data[ , c(-1,-7) ], train_data[ ,7 ] )
#     ↑                                    ↑
# 라벨 제외 훈련데이터    훈련 데이터의 라벨(정답) 

model

# ※ 설명 : tree : 5 ---------> 가지를 5개 만들었다.
# summary(model)

# marry = NO: NO (7)                  # 결혼 안했으면 다 구매 x
# marry = YES:
#   :...car = YES: YES (7/1)             # 결혼을 했는데 차가 있으면 구매 7명인데 1명은 오분류
# car = NO:                                  # 차가 없는 사람중에서
#   :...job = NO: NO (4)                 # 직업이 없으면 다 구매 안했음 ( 4명 )
# job = YES:                              # 직업이 있으면
#   :...age <= 20: NO (2)            # 나이가 20 이하이면 구매 안했음 ( 2명 )
# age > 20: YES (5)              # 나이가 20보다 높으면 구매했음 ( 5명 )


#5. 훈련한 모델로 테스트 데이터 10개를 예측합니다.

result <- predict(model, test_data[  , c(-1,-7) ]   )
result             # [1] NO  NO  YES YES NO 

#6. 모델의 성능(정확도)을 확인합니다.

sum ( result == test_data[  , 7 ] )            # [1] 3 , 5개 중에서 3개 맞춤
sum( result == test_data[  , 7 ] ) / length(result)       # 0.6

#7. 모델의 성능을 높입니다.

library(C50)

model <- C5.0( train_data[ , c(-1,-7) ], train_data[ ,7 ] , trials = 5)
#     ↑                                    ↑
# 라벨 제외 훈련데이터    훈련 데이터의 라벨(정답) 
# ※ 훈련 데이터에서 샘플을 추출해서 5개의 의사결정트리를 만들어서 5개의 의사결정 트리 모델이
# 다수결에 의해서 훈련 데이터를 분류합니다.

model

# Classification Tree
# Number of samples: 25 
# Number of predictors: 5 

# Number of boosting iterations: 5 
# Average tree size: 4.6 



summary(model)

# Trial	    Decision Tree   
# -----	  ----------------  
#   Size      Errors  

# 0	     5    1( 4.0%)
# 1	     3    5(20.0%)
# 2	     5    4(16.0%)
# 3	     5    3(12.0%)
# 4	     5    2( 8.0%)
# boost	          0( 0.0%)   <<
  
  
#   (a)   (b)    <-classified as
# ----  ----
#   14          (a): class NO
# 11    (b): class YES

# 훈련 데이터에 대해서는 정확도 100 % 의 의사결정트리 모델이 나왔습니다.
# trials = 5 를 써서 약한 학습자 5명을 생성해서 5명을 이용해서 강한 학습자를 만들어냄
#  ↓
# 사용자가 직접 알아내야하는 파라미터인 하이퍼 파라미터라고 한다.

result2 <- predict(model, test_data[  , c(-1,-7) ]  )
result2            # [1] NO  NO  YES YES NO

sum( result2 == test_data[ , 7] )             # 3

# 훈련 데이터에 대해서는 100%의 정확도를 보이는 모델이지만 테스트 데이터는
# 5개중에 3만 맞췄습니다. 이런 현상을 과대접합 ( overheating ) 이라고 합니다.

# 30 개는 데이터가 너무 작아서 의사결정트리 + 앙상블을 구현하기 적절하지 않습니다.



