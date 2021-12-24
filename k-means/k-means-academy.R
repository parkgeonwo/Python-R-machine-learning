# ▩ k-means 실습 ( 국영수 점수 데이터 )

# 1. 데이터를 로드합니다.

academy <- read.csv( "academy.csv" )
academy

# 2. 수학/ 영어 점수만 선택합니다.

a2 <- academy[ , c(3,4) ]


# 3. k 값을 4로 주고 비지도학습 시켜 모델 생성

km <- kmeans( a2, 4 )
km

# 4. 학생번호, 수학점수, 영어점수, 분류번호가 같이 출력되게합니다.

result <- cbind( academy[ , c(1,3,4) ], km$cluster )
result

# 1. 영어, 수학 둘다 잘하는 학생
# 2. 영어, 수학 둘다 못하는 학생
3 3. 영어를 잘하는데 수학을 못하는 학생
# 4. 수학은 잘하는데 영어를 못하는 학생

# 5. 시각화를 합니다.

library( factoextra )

fviz_cluster( km, data = a2, stand = F )
