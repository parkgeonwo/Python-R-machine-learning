# 1단계.데이터 수집
wbcd <- read.csv("wisc_bc_data.csv")

nrow(wbcd)                  # 행의 갯수 : 569개
ncol(wbcd)                     # 열의 갯수 : 32개

# 2단계. 데이터 탐색


# 1. 정답에 해당하는 라벨 컬럼의 데이터 분포를 확인합니다.

table(wbcd$diagnosis)

# 
# B   M 
#357 212 


# 2. 이상치를 확인합니다.

install.packages('outliers')
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

colnames(wbcd)                                   # 컬럼명 확인

a <- grubbs.flag( wbcd$radius_mean )           # wbcd 의 radius_mean 컬럼에 이상치가 있는지 확인하기 위해
# 데이터 프레임을 생성
a[ a$Outlier == TRUE,  ]                              # a 데이터 프레임의 Outlier 컬럼이 TRUE 인 데이터를 찾는다.

for ( i in 3:length(colnames(wbcd)) ) {
  a <- grubbs.flag( wbcd[ , colnames(wbcd)[i] ] )
  b <- a[ a$Outlier == TRUE,  'Outlier' ]
  print( paste( colnames(wbcd)[i], '-------->', length(b) ) )
}

# 3. 결측치를 확인합니다.

colSums(is.na(wbcd))


# 4. 히스토그램 그래프 + 정규분포 그래프를 통해서 데이터들이 정규성을 보이는지 확인합니다.

hist(wbcd$dimension_worst)

install.packages("psych")
library(psych)

# 이상치가 많은 컬럼들을 추려서 그려보자

pairs.panels(wbcd[ c('dimension_se', 'symmetry_se', 'perimeter_se') ])        # 히스토그램, 정규분포, 상관계수까지 출력


# ▩ 3단계 : 데이터로 모델 훈련

# 1. 데이터의 구조를 확인하여 라벨( 정답 ) 컬럼이 factor 인지 확인합니다.

str(wbcd)                                 #  $ diagnosis        : chr

wbcd <- read.csv("wisc_bc_data.csv", stringsAsFactors = TRUE)

str(wbcd)                                   #  $ diagnosis        : Factor


# 2. 데이터를 양성과 악성이 잘 섞일 수 있도록 데이터를 섞어줍니다.
# ( 훈련과 테스트 중 한쪽에 양성이 몰리고 악성이 몰릴 수 있으니까 )

sample(10)                       #  [1]  1 10  7  5  4  2  9  8  6  3

wbcd_shuffle <- wbcd[ sample(nrow(wbcd)) ,   ]  
wbcd_shuffle


# 3. 훈련할때 필요한 컬럼만 선택합니다.

wbcd_shuffle2 <- wbcd_shuffle[    ,  c( -1, -2 )  ]                # 첫번째, 두번째 컬럼만 빼기  (환자번호 , 정답 컬럼)

# 4. 컬럼들의 단위가 다 다르므로 데이터를 정규화 합니다.

normalize <- function(x) {
  return (  ( x-min(x) ) / ( max(x) - min(x) )  )
}

wbcd_n <- as.data.frame( lapply(wbcd_shuffle2, normalize) )               # lapply = map 과 같은 역할

summary(wbcd_n)              # 모든 값들이 0 ~1 사이로 바뀜


# 5. 전체 569 개의 행의 데이터를 훈련( 공부 ) 데이터와 테스트 ( 시험 ) 데이터로 나눠줘야 합니다.

# 90 퍼센트는 훈련데이터, 테스트는 10프로

nrow(wbcd_n)           # 569

n_90 <- round( 0.9*nrow(wbcd_n) )
n_90                             # 512

wbcd_train <- wbcd_n[ 1:512 , ]                            # 훈련데이터 구성
wbcd_test <- wbcd_n[ 513:569, ]                           # 테스트 데이터 구성

nrow(wbcd_train)                     # 512    ,    문제지
nrow(wbcd_test)                      # 57      ,     시험

# 6. 정답도 훈련과 테스트로 나눕니다.


wbcd_train_label <- wbcd_shuffle[1:512 , 2]
wbcd_test_label <- wbcd_shuffle[513:569 , 2]

length( wbcd_train_label )              # 512 ,벡터라서 length로 확인
length(wbcd_test_label)                    # 57



#  7. 512 개의 훈련 데이터와 훈련 데이터의 정답으로 거리계산한 데이터로 테스트 데이터 57개를 분류합니다.
#     512로 훈련시키고 57개를 시험쳐보는거 까지 한번에 !!


install.packages("class")                    # knn 구현을 위한 패키지
library(class)

result1 <- knn( train = wbcd_train, test = wbcd_test, cl = wbcd_train_label, k = 1 )
result1


wbcd_test_label


# ▩ 4단계 : 모델 성능 평가

result1 == wbcd_test_label
sum(result1 == wbcd_test_label)
sum(result1 == wbcd_test_label) /57 * 100                     # 98,24561









