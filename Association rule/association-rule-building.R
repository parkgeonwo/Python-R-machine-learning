# ▩ 아프리오리 알고리즘 예제 2  ( 보습학원과 연관성이 높은 업종은 ? )

# " 보습학원이 많은 건물에는 어떤 업종으로 입점해야 장사가 잘될까 ? "

# 1. 데이터를 로드합니다

bd <- read.csv( "building.csv" , header = T )
View ( bd )

# 2. NA 를 0 으로 변경합니다.

bd[ is.na(bd) ] <- 0
bd

# 3. 건물 번호를 제외 시킵니다.

bd2 <- bd[ , -1 ]
bd2

# 4. 데이터 프레임을 행렬로 변환합니다.

bd3 <- as.matrix( bd2, 'Transaction' )       # 행렬로 변환

# 5. 위의 행렬 데이터를 가지고 업종간의 연관 분석을 하시오 !

library(arules)

rules2 <- apriori ( bd3 , parameter = list( supp = 0.2, conf = 0.6, target = 'rules' ) )
rules2         # 46개의 rule 이 발견되었습니다.

# 6. 연관규칙을 확인합니다.

inspect( sort(rules2) )

# 7. 위의 결과에서 보습학원 부분만 따로 떼어서 출력해 봅니다.

rules3 <- subset( rules2 , subset = lhs %pin% '보습학원' & confidence > 0.7 )
inspect( sort( rules3 ) )

# lhs                                rhs             support confidence coverage lift count
# [1] {보습학원}      => {은행}              0.2            1                0.2          5    4    
# [2] {보습학원}      => {카페}              0.2            1                0.2          4    4    
# [3] {보습학원,은행} => {카페}         0.2            1                0.2          4    4    
# [4] {카페,보습학원} => {은행}         0.2            1                0.2          5    4   

# 편의점이 있는 건물에 많은 업종은 무엇인가 ?
  
rules3 <- subset( rules2 , subset = lhs %pin% '편의점' & confidence > 0.7 )
inspect( sort( rules3 ) )

# lhs                                   rhs              support confidence coverage lift     count
# [1]  {편의점}                           => {일반음식점}     0.25    1.0        0.25     2.500000 5    
# [2]  {편의점}                           => {패밀리레스토랑} 0.25    1.0        0.25     2.222222 5    
# [3]  {일반음식점,편의점}                => {패밀리레스토랑} 0.25    1.0        0.25     2.222222 5    
# [4]  {패밀리레스토랑,편의점}            => {일반음식점}     0.25    1.0        0.25     2.500000 5    
# [5]  {편의점}                           => {화장품}         0.20    0.8        0.25     2.666667 4    
# [6]  {편의점,화장품}                    => {일반음식점}     0.20    1.0        0.20     2.500000 4    
# [7]  {일반음식점,편의점}                => {화장품}         0.20    0.8        0.25     2.666667 4    
# [8]  {편의점,화장품}                    => {패밀리레스토랑} 0.20    1.0        0.20     2.222222 4    
# [9]  {패밀리레스토랑,편의점}            => {화장품}         0.20    0.8        0.25     2.666667 4    
# [10] {일반음식점,편의점,화장품}         => {패밀리레스토랑} 0.20    1.0        0.20     2.222222 4    
# [11] {패밀리레스토랑,편의점,화장품}     => {일반음식점}     0.20    1.0        0.20     2.500000 4    
# [12] {일반음식점,패밀리레스토랑,편의점} => {화장품}         0.20    0.8        0.25     2.666667 4 

#  병원이 있는 건물에 가장 연관된 업종은 무엇인가 ?
  
rules3 <- subset( rules2 , subset = lhs %pin% '병원' & confidence > 0.7 )
inspect( sort( rules3 ) )

# lhs                  rhs          support confidence coverage lift     count
# [1] {병원}            => {약국}       0.25    0.8333333  0.30     3.333333 5  

# 보습학원이 있는 건물에 어떤 업종이 많이 있는지 연관분석을 한 결과를 시각화하시오

bd <- read.csv( "building.csv" , header = T )

bd[ is.na(bd) ] <- 0

bd2 <- bd[ , -1 ]

# 데이터 프레임을 이용해서 희소행렬을 출력합니다.

bd3 <- t ( as.matrix( bd2 ) ) %*% as.matrix( bd2 )            # 컬럼을 row 로도 출력을 해줌
bd3

# 시각화합니다.

library(sna)
library(rgl)

bd4 <- bd3 - diag( diag( bd3 ) )
bd4

gplot(bd4 , displaylabel=T , vertex.cex=sqrt(diag(bd3)) , vertex.col = "green" , edge.col="blue" ,
      boxed.labels=F , arrowhead.cex = .3 , label.pos = 3 , edge.lwd = bd4*2) 
