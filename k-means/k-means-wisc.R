# ▩ 유방암 데이터의 악성종양과 양성종양을 k-means 로 2개로 군집화해서 분류

# " 정답 없이 데이터만 보고 분류하는 알고리즘 "

# 1. 데이터를 로드

wisc <- read.csv("wisc_bc_data.csv", header = T)
head(wisc)
ncol(wisc)      # 32

# 2. 필요한 컬럼을 선택합니다.

wisc2 <- wisc[  , 3:32 ]            # 환자 id 와 정답을 제외한 나머지 컬럼들
head(wisc2)

# 3. k-means 모델을 생성합니다.

km <- kmeans( wisc2, 2 )

cbind( wisc$diagnosis, km$cluster )

# 4. 시각화 합니다.

fviz_cluster( km, data = wisc2, stand = F )


# 1은 악성, 2는 양성

# 5. 정확도를 확인하여 성능을 평가합니다.

library(gmodels)

CrossTable( wisc$diagnosis, km$cluster )               # ( 실제, 예상 ) / 코드 실행때마다 1,2 위치가 바뀜



# 정확도 : 486 / 569 = 0.8541     --------------------> 정답없이 이정도면 나쁘지 않은 정확도이다.

# 정답이 없기때문에 성능개선같은건 없다.. 개나줘버려,,
# 정규화하면 좀더 괜찮을수도..? ( 이상치에 대해서 덜 민감해지니까 )

# 이번에는 데이터를 정규화 하고 정확도를 확인하시오 !
  
wisc <- read.csv("wisc_bc_data.csv", header = T)

wisc2 <- wisc_n[  , 3:32 ]            # 환자 id 와 정답을 제외한 나머지 컬럼들

normalize <- function(x) {
  return ( (x-min(x)) / ( max(x)-min(x) ) )
}

wisc_n <- as.data.frame( lapply( wisc2, normalize ) )

km <- kmeans( wisc_n, 2 )

fviz_cluster( km, data = wisc_n, stand = F )



CrossTable( wisc$diagnosis, km$cluster )               # ( 실제, 예상 )

# 정확도 : 528 / 569 = 0.9279 오,,, 많이 올라갔다.

# 위에서는 지금 min max 정규화를 했는데 머신러닝 학습 시킬떄는 scale 함수로 표준화하는것보다
# 정규화를 하는게 더 성능이 좋다고 하는데 진짜로 그런지 실험하세요 

# 표쥰화 : 평균이 0이고 표준편차가 1인 데이터로 변경 ( 함수 : scale )
# 정규화 : 데이터를 0 ~ 1 사이로 변경 ( normalize )

wisc <- read.csv("wisc_bc_data.csv", header = T)

wisc2 <- wisc_n[  , 3:32 ]            # 환자 id 와 정답을 제외한 나머지 컬럼들

wisc_s <- as.data.frame( lapply( wisc2, scale ) )

km <- kmeans( wisc_s, 2 )

fviz_cluster( km, data = wisc_s, stand = F )



CrossTable( wisc$diagnosis, km$cluster )               # ( 실제, 예상 )

# 정확도 : 515 / 569 = 0.9051  
