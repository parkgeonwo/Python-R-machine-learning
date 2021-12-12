wine <- read.csv("wine.csv", stringsAsFactors = TRUE) 
# nrow(wine)               # 178
# ncol(wine)              # 14

# table(wine$Type)

# t1 t2 t3 
# 59 71 48 

set.seed(1)
wine_shuffle <- wine[ sample(nrow(wine)) ,   ]  

wine_shuffle2 <- wine_shuffle[    ,  -1  ]                # 첫번째 컬럼만 빼기  (정답 컬럼)

normalize <- function(x) {
  return (  ( x-min(x) ) / ( max(x) - min(x) )  )
}

wine_n <- as.data.frame( lapply(wine_shuffle2, normalize) )               # lapply = map 과 같은 역할


# n_90 <- round( 0.9*nrow(wine_n) )                 # 160

wine_train <- wine_n[ 1:160 , ]                            # 훈련데이터 구성
wine_test <- wine_n[ 161:178, ]                           # 테스트 데이터 구성

# nrow(wine_train)         # 160
# nrow(wine_test)            # 18

wine_train_label <- wine_shuffle[1:160 , 1]
wine_test_label <- wine_shuffle[161:178 , 1]

# length(wine_train_label)         # 160
# length(wine_test_label)          # 18

library(class)

result1 <- knn( train = wine_train, test = wine_test, cl = wine_train_label, k = 1 )

sum(result1 == wine_test_label) /18 * 100                     # 94.44444


# 여러 k 값에서 정확도 알아보기 

for (i in c(seq(1,150,2))){
  result1 <- knn(wine_train, wine_test, cl=wine_train_label, k=i)
  a <- sum(result1 == wine_test_label)/length(wine_test_label)*100
  print(paste(i,' --------> ',a))  }


# 이원교차표 확인

library(gmodels)

CrossTable( x= wine_test_label, y = result1, prop.chisq = FALSE )




