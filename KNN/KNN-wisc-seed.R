wbcd <- read.csv("wisc_bc_data.csv", stringsAsFactors = TRUE) 

set.seed(1)
wbcd_shuffle <- wbcd[ sample(nrow(wbcd)) ,   ]  

wbcd_shuffle2 <- wbcd_shuffle[    ,  c( -1, -2 )  ]                # 첫번째, 두번째 컬럼만 빼기  (환자번호 , 정답 컬럼)

normalize <- function(x) {
  return (  ( x-min(x) ) / ( max(x) - min(x) )  )
}

wbcd_n <- as.data.frame( lapply(wbcd_shuffle2, normalize) )               # lapply = map 과 같은 역할

wbcd_train <- wbcd_n[ 1:512 , ]                            # 훈련데이터 구성
wbcd_test <- wbcd_n[ 513:569, ]                           # 테스트 데이터 구성

wbcd_train_label <- wbcd_shuffle[1:512 , 2]
wbcd_test_label <- wbcd_shuffle[513:569 , 2]

library(class)

result1 <- knn( train = wbcd_train, test = wbcd_test, cl = wbcd_train_label, k = 1 )

sum(result1 == wbcd_test_label) /57 * 100                     # 96.49123

temp <- c()

# 정확도를 더 올리기 위한 k 값을 알아내시오 ~

for (i in c(seq(1,300,2))) {
  result1 <- knn( train = wbcd_train, test = wbcd_test, cl = wbcd_train_label, k = i )
  
  temp <- append(temp, sum(result1 == wbcd_test_label) /57 * 100)
}

temp

plot( temp, type = 'o', col = 'blue', ylim = c (0,105) )  

