# 직업과 성별과 결혼 유무등의 데이터를 가지고 영화장르를 예측하는데
# 나이브 베이즈 모델을 생성하시오 ! ( 데이터 셋 : movie.csv )

# 1. 데이터를 로드합니다.

movie <- read.csv("movie.csv", stringsAsFactor = TRUE)
str(movie)         # all factor

# 맨앞에 있는 type 이 라벨(정답) 입니다.

table(movie$장르)

#  SF   공포 로맨틱   무협 스릴러   액션 코미디 
#  5      5      9      5      5      5      5 


prop.table(table( movie$장르 ))                    #  비율을 알수있음.

#      SF      공포    로맨틱      무협    스릴러      액션    코미디 
# 0.1282051 0.1282051 0.2307692 0.1282051 0.1282051 0.1282051 0.1282051

dim(mush)                 # [1] [1] 39  6 , 전체 건수가 어떻게 되는지 확인

# 2. 결측치를 확인합니다.

colSums(is.na(movie))                    # 모두 0 이다.

# 3. 이상치를 확인합니다.

# 명목형 데이터여서 이상치를 확인할 필요가 없습니다.

# 4. 명목형 데이터가 있는지 확인합니다.

# 전부 명목형 데이터 입니다.

# 5. 데이터를 정규화 합니다.

# 전부 명목형 데이터 이므로 정규화 작업도 필요하지 않습니다.

# 6. 훈련 데이터와 테스트 데이터를 분리합니다.

# 데이터 shuffle 과 데이터 분리를 효율적이면서도 편하게 수행할 수 있는 패키지를 이용해서 분리해보겠습니다.

install.packages("caret")
library(caret)

set.seed(1)
k <- createDataPartition( movie$장르, p = 0.8 , list = F )   # 훈련데이터 80 %, 테스트 20 %  / sample 사용하지 않고 쉽게 만듦
# 또한, 리스트 형태로 만들지 말아라
k           # 훈련데이터의 index가 나옴

train_data <- movie[ k, ]
test_data <- movie[ -k,  ]

dim(train_data)           # [1] 32  6
dim(test_data)           # [1] 7 6

prop.table( table(train_data$장르) ) 

#    SF   공포 로맨틱   무협 스릴러   액션 코미디 
#  0.125  0.125  0.250  0.125  0.125  0.125  0.125 

prop.table( table(test_data$장르) )

#        SF      공포    로맨틱      무협    스릴러      액션    코미디 
# 0.1428571 0.1428571 0.1428571 0.1428571 0.1428571 0.1428571 0.1428571

# ※ 훈련데이터와 테스트 데이터의 독버섯과 정상버섯이 거의 50:50으로 균등하게 분포되어있습니다.
# sample에 비해서 데이터가 더 훈련하고 테스트하기 쉽게 잘 나눠져서 좋다.


# 7. 나이브 베이즈 모델을 생성합니다.

install.packages("e1071")
library(e1071)

model <- naiveBayes( type~ . , data= train_data )        # type 는 라벨 ,  '.'은 라벨외의 모든컬럼을 뜻함
model


# cap_shape = 버섯 cap의 모양
# cap_surface = 버섯 cap의 표면,, 등등

# 버섯 데이터로 빈도표를 만들고서 우도표를 생성했다.

# 8. 훈련 데이터와 라벨( 정답 )으로 모델을 훈련시킵니다.

# 7번에서 다 수행했습니다.


# 9. 훈련된 모델로 테스트 데이터를 예측합니다.

result <- predict( model, test_data[  , -1] )              # 정답 컬럼 빼고

# 테스트 데이터의 정답을 제외하고 예측합니다.
result

# 10. 모델의 성능을 평가합니다.

sum ( (result == test_data[, 1]) / length(test_data[ , 1]) )         # 

# 어제는 유방암 데이터가 전부 숫자여서 knn 알고리즘을 이용해서 기계학습 시켰고
# 오늘은 독버섯 데이터가 전부 명목형이어서 naivebayes 를 이용해서 기계학습 시켰습니다.


# 11. 모델의 성능을 높입니다.

model2 <- naiveBayes( type~ . , data = train_data, laplace = 0.0001 )

result2 <- predict( model2, test_data[  , -1] )

sum ( (result2 == test_data[, 1]) / length(test_data[ , 1]) )          # 

