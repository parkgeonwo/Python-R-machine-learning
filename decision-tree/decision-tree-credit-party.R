# 1. party 패키지를 설치합니다.

install.packages('party')
library(party)

# 2. 독일 은행 데이터를 불러옵니다.

credit <- read.csv("credit.csv", stringsAsFactors = T )
nrow(credit)      # 1000

# 3. 훈련 데이터와 테스트 데이터로 분리합니다.

train_num <- createDataPartition ( credit$default, p = 0.9 , list = F )

train_data <- credit[ train_num,  ]
test_data <- credit[ -train_num,  ]

nrow(train_data)       # 900
nrow(test_data)        # 100

# 4. 모델 생성

# C5.0 패키지 문법 : model <- C50( 라벨을 뺀 데이터, 라벨 데이터 )
# party 패키지 문법 : model <- ctree( 라벨~. , data = 훈련 데이터 프레임명 )

model <- ctree( default~. , data = train_data )        # '.' 은 나머지 모든컬럼을 말한다. / 여기서는 라벨을 뺀 모든컬럼

# 5. 모델 예측

result <- predict ( model, test_data[ , -17 ]  )
table(result)

#   result
#  no yes 
#  89  11 

# 6. 모델 평가

table( test_data[  , 17], result )

#      result
#       no yes
#   no  66   4
#  yes 23   7

# ※ 73 % 의 정확도를 보이는 모델이 생성되었습니다.
# FN 값은 23으로 높다..
# 카이제곱으로 해도 크게 개선되지 않았다. 다른 알고리즘 모델을 써야한다.

# 7. 모델 시각화

plot(model)
