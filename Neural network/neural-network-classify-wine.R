# ▩ 인공신경망을 이용해서 분류하기 ( R )

# " 와인의 등급을 분류하는 신경망 "

# 1. 데이터를 로드합니다.

wine <- read.csv("wine.csv", stringsAsFactors = T)         # 첫번째 컬럼 'Type'가 라벨
nrow(wine)               # 178
ncol(wine)                 # 14
unique(wine$Type)       # t1, t2, t3
str(wine)                # Type 만 문자

# 2. 결측치가 있는지 확인합니다.
colSums(is.na(wine))

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

wine <- read.csv("wine.csv")

for (i in 2:length(colnames(wine))){
  
  a = grubbs.flag(wine[,colnames(wine)[i]])
  b = a[a$Outlier==TRUE,"Outlier"]
  print ( paste( colnames(wine)[i] , '--> ',  length(b) )  )
  
}

# 이상치가 거의없는 좋은 데이터

# 4. 정규화를 진행합니다.

normalize <- function (x) {
  return ( ( x - min(x) ) / ( max(x) - min(x) ) )
}

wine_n <- as.data.frame( lapply( wine[  , -1 ], normalize ) )            # wine의 1열 빼고 정규화
summary(wine_n)

wine2 <- cbind( Type = wine$Type, wine_n )           # wine의 Type 열과 정규화한 wine_n 합쳐줌, 열이름 Type으로

head(wine2)

# 5. 훈련데이터와 테스트 데이터를 분리합니다.

library(caret)

set.seed(1)
train_num <- createDataPartition( wine2$Type, p = 0.9 , list = FALSE )
train_data <- wine2[ train_num ,  ]
test_data <- wine2[ -train_num ,  ]

nrow(train_data)        # 162
nrow(test_data)        # 16

# 6. 모델을 설정합니다.

install.packages( "nnet" )
library(nnet)

# 7. 모델을 훈련시킵니다.

set.seed(1)
model <- nnet( Type ~. , data = train_data , size = 2 )

# 설명 : size =2 는 은칙층 1개로 하고 은닉층의 뉴런의 갯수를 2개로 하겠다는 뜻입니다.



# 8. 테스트 데이터를 예측합니다.

result <- predict ( model, test_data, type = 'class' )
result

# 9. 모델을 평가합니다.

sum( result == test_data[  , 1] ) / length( test_data[  , 1 ] )              # 1

# 10. 모델의 성능을 높입니다.

model2 <- nnet( Type~. , data = test_data, size = 3 )       # 은닉 1층은 뉴런 3개
result2 <- predict( model2, test_data, type = 'class' )
sum( result2 == test_data[  , 1] ) / length( test_data[  , 1 ] )         # 1
