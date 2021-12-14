# 1. 데이터 로드

credit <- read.csv("credit.csv", stringsAsFactors = TRUE)
head(credit)
str(credit)

p 207 의 데이터 설명 : 독일의 한 신용기관에서 얻은 대출 정보가 있는 데이터

# 정답( 라벨 ) 컬럼 : default  ----------> yes : 대출금 상환 안함
# no : 대출금 상환 함
# prop.table( table ( credit$default ) )

# no yes 
# 0.7 0.3

# 30 % 의 해당하는 사람들이 대출금을 상환하지 않고 있음
# 머신러닝 모델을 


# 2. 데이터 탐색

# checking_balance : 예금계좌
# saving_balance : 적금계좌

# amount : 대출금액 ( 250 마르크 ~ 18424 마르크 )
# 100 마르크가 우리나라돈으로 6~7만원

# amount 의 데이터를 히스토그램 그래프로 그리시오 

hist(credit$amount)

summary(credit$amount)

#   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#    250    1366    2320    3271    3972   18424 
# 설명 : 최소 250 마르크 ( 1750 만원 ) ~ 18424 마르크 ( 약 12억 8천만원 ) 사이로 구성되어 있다.

# 예금 계좌에 입금된 돈의 분포를 확인하시오 !
table( credit$checking_balance )

#    < 0 DM   > 200 DM 1 - 200 DM    unknown 
#          274               63                269         394 
# 1000개의 고객 계좌중에서 200 마르크 이상의 계좌가 63개
# 아예 계좌가 없는 고객이 274 계좌, 1 ~ 200 마르크 사이가 269 명이 있습니다.



# 3. 훈련과 테스트로 데이터를 분리 ( 훈련 데이터 : 9, 테스트 데이터 : 1 )

library(caret)
set.seed(1)          # 어느 자리에서든 동일한 방법으로 훈련과 테스트 데이터를 분리하기 위해서
train_num <- createDataPartition( credit$default, p = 0.9, list = F )
train_num           # 1000개의 데이터 중에 90 %에 해당하는 데이터를 샘플링한 인덱스 번호

train_data <- credit[ train_num,  ]
test_data <- credit[-train_num, ]

nrow(train_data)       # 900
nrow(test_data)        # 100

# 4. 훈련 데이터로 모델을 생성합니다.

library(C50)           # 엔트로피 지수를 이용해서 순수도를 구하고 분류하는 패키지

# 5. 훈련된 모델을 테스트 데이터를 예측합니다.

# 문법 : model <- C5.0( 라벨을 뺀 나머지 데이터, 라벨 컬럼 데이터 )
# ncol( train_data )        # 17


model <- C5.0( train_data[ , -17 ], train_data[, 17] )

# ※ 설명 : 900 개의 훈련 데이터로 학습한 모델을 생성했습니다.

summary(model)

# checking_balance = unknown: no (356/42)
# checking_balance in {< 0 DM,> 200 DM,1 - 200 DM}:
#   :...amount > 8648: yes (31/6)
# amount <= 8648:
#   :...credit_history in {perfect,very good}:
#   :...housing in {other,rent}: yes (26/3)
# :   housing = own:
#   :   :...savings_balance in {> 1000 DM,500 - 1000 DM,
#     :       :                   unknown}: no (8/2)
# :       savings_balance = 100 - 500 DM:
#   :       :...months_loan_duration <= 16: yes (3)
# :       :   months_loan_duration > 16: no (3)
# :       savings_balance = < 100 DM:
#   :       :...age > 33: yes (8)

# ※ 설명 : checking_balance (예금계좌) 에 200마르크 이상의 돈이 있는 사람들 중에서
# 대출금액이 8648 보다 큰 31명의 사람들의 대출금을 상환하지 않았고
# 대출금액이 8648 보다 작은 사람들중에 집이 월세인 사람들 26명이 대출금을 상환하지 않았다.
# 집이 자가소유인 8명은 대출금을 상환했습니다.
# 집이 자가소유이면서 적금계좌에 500마르크가 있는 사람들중에
# 적금을 부은 개월수가 16개월 이상이면 대출금을 상환했고,
# 16개월 보다 작으면 상환하지 않았다.
# 이렇게 분석해서 글을 보여주면 고객이 아주 좋아한다!!

# 문법 : result <- predict ( 모델, 라벨을 뺀 테스트 데이터 )

result <- predict ( model, test_data[ , -17 ] )
table(result)

#  result
#  no yes 
#  81  19 
# 테스트 100명에 대해서 81명은 대출금을 상환했고 19명은 상환하지 않았다고 예측하고 있습니다.

# 6. 모델의 성능을 평가합니다.

# 문법 : table( 실제값, 예측값 )

table( test_data[  ,17 ], result )

#    result         실제
#                       no yes         
#    예측    no  59  11
#                 yes 22   8


# ※ 100 명중에 67명을 정확하게 예측했으므로 정확도는 67 % 모델입니다.
# 채무이행할거 예측했는데 채무를 불이행한 FN 값이 22명이나 되는 모델이므로 성능개선이 필요합니다.

library(gmodels)
CrossTable(test_data[,17], result)


# 7. 모델의 성능을 개선합니다.

# 의사결정트리의 성능을 높이려면 trials 의 갯수를 조정합니다.
# trials 는 의사결정 나무의 갯수를 결정하는 하이퍼 파라미터 입니다.

model2 <- C5.0( train_data[ ,-17 ], train_data[ , 17], trials = 100 )

result2 <- predict( model2, test_data[ , -17] )

table(test_data[ , 17], result2)

#        result2
#        no yes
#  no  60  10
#  yes 20  10

# ※ trials 를 100으로 지정했더니 FN 값이 2가 줄었습니다.
