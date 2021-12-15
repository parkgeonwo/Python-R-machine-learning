# 1. 데이터를 로드합니다.

mush <- read.csv("mushrooms.csv", stringsAsFactors = T)

# 2. 결측치 확인

colSums(is.na(mush))

# 3. 훈련데이터와 테스트 데이터로 분리합니다.

library(caret)
set.seed(1)
train_num <- createDataPartition( mush$type, p = 0.8 , list = F )

train_data <- mush[train_num, ]         # 훈련데이터 80% 구현
test_data <- mush[-train_num, ]         # 테스트데이터 20% 구현

nrow(train_data)      # 6500
nrow(test_data)        # 1624

# 4. 훈련데이터로 모델을 훈련시킵니다.

install.packages("RWeka")
library(RWeka)

model2 <- JRip( type~. , data = train_data )
model2                                                                  # 책 245p 에 아래의 내용에 대한 해석이 나옵니다.

# (odor = f) => type=p (1732.0/0.0)
# (gill_size = n) and (gill_color = b) => type=p (921.0/0.0)
# (gill_size = n) and (odor = p) => type=p (205.0/0.0)
# (odor = c) => type=p (155.0/0.0)
# (spore_print_color = r) => type=p (52.0/0.0)                 # 버섯 머리 아래쪽을 종이에 찍은 색
# (stalk_surface_above_ring = k) and (gill_spacing = c) => type=p (58.0/0.0)
# (habitat = l) and (cap_color = w) => type=p (7.0/0.0)
# (stalk_color_above_ring = y) => type=p (3.0/0.0)
# => type=e (3367.0/0.0)

summary(model2)

# === Confusion Matrix ===
  
#   a    b   <-- classified as
# 3367    0 |    a = e
# 0 3133 |    b = p

# ※ 설명 : 위의 작은 이원교차표에서 훈련데이터에 대해서 100% 정확도를 보여주는 결과가 나타남

# 5. 훈련된 모델에 테스트 데이터를 넣어서 예측을 합니다.

result2 <- predict( model2, test_data[  , -1] )
result2

# 6. 모델을 평가합니다.

sum(result2 == test_data[ , 1]) / nrow(test_data)            #  1 , length(result2) 로 해도 같은 결과

# ※ 설명 : 나이브 베이즈 모델  : 정확도 ?
#   OneR 알고리즘 : 98 %
#   Riper 알고리즘 : 100%


