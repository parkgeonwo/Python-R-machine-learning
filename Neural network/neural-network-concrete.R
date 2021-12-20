# ▩ 신경망 실습 1 ( 콘크리트 강도를 예측하는 신경망 만들기 ) p 328

# " 어떻게 조합해야 강도가 높은 콘크리트를 만들 수 있는가 ? "

# " 콘크리트의 강도를 예측하는 신경망 만드는 실습 "

# 자갈, 모래, 시멘트 등을 몇대몇 비율로 섞었을때 어느정도 강도가 나오는지 예측하는
# 신경망

# 신경망으로 분류도 할 수 있고 수치예측도 할 수 있는데 이번 실습은 수치예측입니다.

# *콘크리트 데이터 소개( concrete.csv )

# 1. mount of cement : 콘크리트의 총량
# 2. slag : 시멘트
# 3. ash : 분(시멘트)
# 4. water : 물
# 5. superplasticizer : 고성능 감수제 (콘크리트의 강도를 높이는 첨가제)
# 6. coarse aggregate : 굵은 자갈
# 7. fine aggreagate : 잔 자갈
# 8. aging time : 숙성 시간
# 9. strengh : 강도 ( 정답라벨 )

# 1. 데이터를 로드합니다.
concrete <- read.csv("concrete.csv")
str(concrete)

nrow(concrete)      # 1030
ncol(concrete)       # 9

# 2. 결측치가 있는지 확인합니다.
colSums(is.na(concrete))

# 3. 이상치가 있는지 확인합니다.

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

wisc <- read.csv("concrete.csv")

for (i in 1:length(colnames(wisc))){
  
  a = grubbs.flag(wisc[,colnames(wisc)[i]])
  b = a[a$Outlier==TRUE,"Outlier"]
  print ( paste( colnames(wisc)[i] , '--> ',  length(b) )  )
  
}

# [1] "superplastic -->  5"
# [1] "age -->  59"


# 4. 데이터를 정규화합니다.

normalize <- function (x) { return ( ( x - min(x) ) / ( max(x) - min(x) ) ) }

concrete_norm <- as.data.frame( lapply(concrete, normalize) )
summary(concrete_norm)

# 5. 훈련 데이터와 테스트 데이터로 분리합니다 ( 8:2 )

library(caret)
set.seed(1)
k <- createDataPartition( concrete_norm$strength, p = 0.8, list = F )

train_data <- concrete_norm[ k , ]
test_data <- concrete_norm[ -k , ]

nrow(train_data)       # 826
nrow(test_data)          # 204

# 6. 모델 생성

install.packages("neuralnet")
library(neuralnet)

# 7. 훈련 데이터로 모델생성

concrete_model <- neuralnet( formula = strength ~ cement + slag + ash + water + superplastic + coarseagg +
                               + fineagg + age, data = train_data )

# 신경망 시각화

plot( concrete_model )


# 0층                                1층             2층                  >>>> 2층 신경망( 다층신경망 )


# 8. 테스트 데이터를 예측

result <- compute( concrete_model , test_data[ , 1:8 ]  )
result$net.result

# 9. 모델 성능 평가

cor( result$net.result , test_data[  , 9] )        # 0.8245659

# ※ 설명 : 분류이면 이원교차표를 통해서 정확도를 확인할텐데 수치예측이므로 상관계수로
# 모델의 성능을 체크해야 합니다.

# 10. 모델 성능 개선

# 위의 신경망은 2층 신경망이었습니다. 그래서 계산방법은 3층으로 신경망을 늘리고
# 은닉층의 뉴런의 갯수도 1개에서 7개로 늘려봅니다.



concrete_model2 <- neuralnet( formula = strength ~ cement + slag + ash + water + superplastic + coarseagg +
                                + fineagg + age, data = train_data, hidden = c(5,2) )

# ※ 설명 : hidden = c(5,2)                                           ## 하이퍼 파라미터
                     # ↓↓
# 은닉 1층의 뉴런수, 은닉2층의 뉴런수 

plot( concrete_model2 )



result2 <- compute( concrete_model2, test_data[ , 1:8 ] )
cor(result2$net.result, test_data[ , 9  ] )              # 0.927863

# 위의 신경망의 성능을 더 올리시오 !
  
  set.seed(1)      # 항상 일정한 상관계수를 보기 위해서 모델 생성전에 설정합니다.
concrete_model3 <- neuralnet( formula = strength ~ cement + slag + ash + water + superplastic + coarseagg +
                                + fineagg + age, data = train_data, hidden = c(5,5,2) )

result3 <- compute( concrete_model3, test_data[ , 1:8 ] )
cor(result3$net.result, test_data[ , 9  ] )              #  0.9408049

plot(concrete_model3)



# hidden(5,5,5,2) 로 하면 오히려 떨어진다. ( 사공이 많아서 오히려 결과가 나빠짐 ) 