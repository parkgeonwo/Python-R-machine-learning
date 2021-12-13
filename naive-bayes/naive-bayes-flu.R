# 독감 데이터로 나이브 베이즈 모델로 생성해서 독감환자인지 아닌지 분류하는 모델을 만드시오
# ( flu.csv )

# patient_id  : 환자번호
# chills : 오한
# runny_nose : 콧물
# headache : 두통
# fever : 열
# flue : 독감여부



flu <- read.csv("flu.csv", stringsAsFactor = TRUE)

set.seed(1)
k <- createDataPartition( flu$flue, p = 0.8 , list = F )   # 훈련데이터 80 %, 테스트 20 %  / sample 사용하지 않고 쉽게 만듦
# 또한, 리스트 형태로 만들지 말아라
train_data <- flu[ k, ]
test_data <- flu[ -k,  ]
model <- naiveBayes( flue~ . , data= train_data )        # type 는 라벨 ,  '.'은 라벨외의 모든컬럼을 뜻함
result <- predict( model, test_data[  , -6] )              # 정답 컬럼 빼고
sum ( (result == test_data[, 6]) / length(test_data[ , 6]) )         # [1]  1

model2 <- naiveBayes( flue~ . , data = train_data, laplace = 0.0001 )
result2 <- predict( model2, test_data[  , -6] )
sum ( (result2 == test_data[, 6]) / length(test_data[ , 6]) )          #  [1] 1
