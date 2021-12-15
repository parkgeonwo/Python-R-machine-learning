# 1.버섯 데이터 로드한다.
# 2. 결측치를 확인합니다.
# 3. 이상치를 확인합니다.
# 4. 훈련 데이터와 테스트 데이터로 나눕니다.
# 5. 정규화 작업 수행
# 6. 훈련 데이터로 모델을 훈련시킵니다.
# 7. 훈련된 모델에 테스트 데이터를 넣어서 예측합니다.
# 8. 모델을 평가합니다.


# 1.버섯 데이터 로드한다.

mush <- read.csv("mushrooms.csv", stringsAsFactors = T)

# 데이터 설명 : UCI 머신러닝 저장소에서 제공하는 데이터이며 23종의 버섯과
# 8124개의 버섯샘플에 대한 정보가 포함되어 있습니다.
# 버섯샘플 22개의 특징은 갓모양, 갓색깔, 냄새, 주름크기, 주름색, 줄기모양, 서식지와 같은
# 특징이 있습니다.

# 2. 결측치를 확인합니다.

colSums(is.na(mush))

# 3. 이상치를 확인합니다.

# 전부 명목형 데이터 입니다.

# 4. 훈련 데이터와 테스트 데이터로 나눕니다.

library(caret)
set.seed(1)
train_num <- createDataPartition( mush$type, p = 0.8 , list = F )

train_data <- mush[train_num, ]         # 훈련데이터 80% 구현
test_data <- mush[-train_num, ]         # 테스트데이터 20% 구현

nrow(train_data)      # 6500
nrow(test_data)        # 1624

# 5. 정규화 작업 수행

# 명목형 데이터 이므로 정규화 작업을 할 필요가 없습니다.

# 6. 훈련 데이터로 모델을 훈련시킵니다. ( oneR 알고리즘 )

install.packages('OneR')    
library(OneR)                      # 한가지 조건만 가지고 분류하는 알고리즘
# 장점 : 간단하다, 단점 : 정확하게 분류하지 못한다.

model <- OneR( type~. , data = train_data )

# 문법 : model <- OneR(정답컬럼~모든컬럼 , data = 데이터 프레임 )

model               # 알고리즘이 데이터에서 발견한 패턴을 확인할 수 있습니다.

# Rules:                                            # 버섯 냄새 한가지만 가지고 다음과 같이 분류했습니다.
#   If odor = a then type = e
# If odor = c then type = p
# If odor = f then type = p
# If odor = l then type = e
# If odor = m then type = p
# If odor = n then type = e
# If odor = p then type = p
# If odor = s then type = p
# If odor = y then type = p

summary(model)

# Pearson's Chi-squared test:
# X-squared = 6151.1, df = 8, p-value < 2.2e-16          # 2.2 * 0.00..001 ( 소수점아래 0이 16개 )

# 귀무가설 : 냄새로 독버섯과 정상버섯을 분류할 수 없다.
# 대립가설 : 냄새로 독버섯과 정상버섯을 분류할 수 있다.

# p-value 값이 2.2e-16 으로 매우 작으므로 대립가설을 채택할 충분할 근거가 있다.


# 7. 훈련된 모델에 테스트 데이터를 넣어서 예측합니다.

result <- predict( model, test_data[ , -1] )           # 라벨 컬럼뺀 test_data
result


# 8. 모델을 평가합니다.

sum( test_data[  ,1 ] == result ) / nrow(test_data)       # [1] 0.9815271

# 9. 이원교차표를 확인해서 FN 값이 몇개가 있는지 확인합니다.

library(gmodels)

CrossTable(test_data[ , 1], result)

# ※ 설명 : 정확도는 98 % 이나 FN 값이 높아서 FN 값을 줄일 수 있도록 개선할 필요가 있습니다.
