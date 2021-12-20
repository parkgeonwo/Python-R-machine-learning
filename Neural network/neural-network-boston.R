▩ boston 집값을 예측하는 인공신경망 모델 만들기

# 1. 데이터를 로드합니다.
boston <- read.csv("boston.csv")

# 2. 결측치를 확인합니다.
colSums(is.na(boston))

# 3. 이상치를 확인합니다.

library(outliers)

grubbs.flag <- function(x) {
  outliers <- NULL
  test <- x
  grubbs.result <- grubbs.test(test)
  pv <- grubbs.result$p.value
  while(pv < 0.05) {
    outliers <- c(outliers,as.numeric(strsplit(grubbs.result$alternative," ")[[1]][3]))
    test <- x[!x %in% outliers]
    grubbs.result <- grubbs.test(test)
    pv <- grubbs.result$p.value
  }
  return(data.frame(X=x,Outlier=(x %in% outliers)))
}

wisc <- read.csv("boston.csv")

for (i in 1:length(colnames(wisc))){
  
  a = grubbs.flag(wisc[,colnames(wisc)[i]])
  b = a[a$Outlier==TRUE,"Outlier"]
  print ( paste( colnames(wisc)[i] , '--> ',  length(b) )  )
  
}

# [1] "CRIM -->  24"
# [1] "ZN -->  1"
# [1] "B -->  98"


# 4. 데이터를 정규화 시킵니다.

normalize <- function (x) { return ( ( x - min(x) ) / ( max(x) - min(x) ) ) }

boston_norm <- as.data.frame( lapply(boston, normalize) )
summary(boston_norm)

# 5. 훈련데이터로 테스트 데이터로 데이터로 분리합니다.

library(caret)
set.seed(1)
k <- createDataPartition( boston_norm$price, p = 0.9, list = F )

train_data <- boston_norm[ k , ]
test_data <- boston_norm[ -k , ]

nrow(train_data)       # 458
nrow(test_data)          # 48


# 6. 인공신경망 모델을 생성합니다.

library(neuralnet)

# 7. 훈련 데이터로 인공신경망 모뎅을 학습 시킵니다.

set.seed(1)
boston_model <- neuralnet( formula = price ~  CRIM + ZN + INDUS + CHAS + NOX +
                             + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT, data = train_data )

# 신경망 시각화

plot( boston_model )



# 8. 테스트 데이터를 예측합니다.

library(neuralnet)
boston_result <- neuralnet::compute( boston_model , test_data[ , 2:14 ]  )        # dplyr 에 compute가 있음
boston_result$net.result

# ※ 설명 : compute 가 dplyr 패키지에도 compute가 있어서 neuralnet 과 충돌이 있어서
# 그냥 compute 만 했을때 자꾸 실행안되면서 오류가 나서 neuralnet:compute 해줘야
# neuralnet의 compute를 써라 ~ 라고 명령을 하는 것입니다.

# 9. 모델 성능 평가

cor( boston_result$net.result , test_data[  , 15] )        # 0.8312131

# -----------------------------------------------------------------------------------------------------------------------------
# 10. 모델의 성능을 더 올립니다. ( 하이퍼파라미터 조절로 올리는 시도 )
  
  set.seed(1)
boston_model2 <- neuralnet( formula = price ~  CRIM + ZN + INDUS + CHAS + NOX +
                              + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT, data = train_data
                            , hidden = c(5,2) )

boston_result2 <- neuralnet::compute( boston_model2 , test_data[ , 2:14 ]  )        
cor( boston_result2$net.result , test_data[  , 15] )        #  0.7838629

# seed 값 변화 주면서 바꿔도 됨

set.seed(1) , hidden = c(5,5,2)       # 0.91805
set.seed(1) , hidden = c( 11,6,2 )    # 0.94

# 11. 모델의 성능을 더 올립니다. ( 파생변수 추가로 올리는 시도 )


#####################################################################
#  위에서 만들었던 파생변수 5개를 하나씩만 사용해서 파생변수 추가로 
# 모델의 성능을 더 올릴수 있는지 확인하시오 !
# 파생변수는 데이터로드하고 바로 추가해야한다.

boston <- read.csv("boston.csv")

boston$rm_grade <- ntile( boston$RM , 4 ) 

normalize <- function (x) { return ( ( x - min(x) ) / ( max(x) - min(x) ) ) }

boston_norm <- as.data.frame( lapply(boston, normalize) )

library(caret)
set.seed(1)
k <- createDataPartition( boston_norm$price, p = 0.9, list = F )

train_data <- boston_norm[ k , ]
test_data <- boston_norm[ -k , ]

library(neuralnet)
set.seed(1)
boston_model <- neuralnet( formula = price ~  CRIM + ZN + INDUS + CHAS + NOX +
                             + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT + rm_grade, data = train_data )

boston_result <- neuralnet::compute( boston_model , test_data[ , c(-1,-15) ]  )        
cor( boston_result$net.result , test_data[  , 15] )       


boston$rm_grade <- ntile( boston$RM , 4 )          # 0.8231838
boston$lstat_grade <- ntile( boston$LSTAT , 4 )    # 0.84513
boston$age_grade <- ntile( boston$AGE , 4 )          # 0.8257861
boston$indus_grade <- ntile( boston$INDUS , 4 )   # 0.8121444
boston$nox_grade <- ntile( boston$NOX , 4 )        # 0.8226917
