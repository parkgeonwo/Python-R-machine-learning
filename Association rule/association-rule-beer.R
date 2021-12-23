# ▩ 아프리오리 알고리즘 예제 1. ( 맥주와 기저기 )

# 금요일 밤에 남자들이 기저귀를 사러 마트에가면 맥주를 같이 사는 패턴을 발견

# 1. 데이터를 로드합니다.

x <- data.frame( beer = c(0,1,1,1,0), bread = c(1,1,0,1,1) , cola = c(0,0,1,0,1) , diapers = c(0,1,1,1,1),
                 eggs = c(0,1,0,0,0), milk = c(1,0,1,1,1) )

# 2. arules 패키지를 설치 ( 연관규칙 구현하는 패키지 )

install.packages('arules')
library(arules)

# 3. x 데이터 프레임을 행렬로 변환합니다. ( arules 가 데이터를 행렬로 제공받기 때문입니다. )

x2 <- as.matrix(x, 'Transaction')

# 4. arules 패키지의 apriori 함수를 이용해서 연관관계를 분석합니다.

rules1 <- apriori( x2, parameter = list( supp = 0.2 , conf = 0.6, target = 'rules' ) )

# ※ 설명 : x2 데이터에서 지지도가 0.2 이상이고 신뢰도가 0.6 이상인 rule을 발견해라

rules1                      # set of 49 rules = 49 개의 연관품목들을 발견했습니다.

# 5. 연관품목들 확인하는 방법

inspect( sort(rules1) )

# 결과 설명 : 맥주를 샀을때 기저귀를 살 연관성이 가장 높고 그 다음으로는 기저귀를 샀을 때
맥주를 살 연관성이 두번째로 높습니다.
3개의 조합으로는 맥주, 빵, 기저귀가 가장 연관성이 높습니다.

# 6. 위의 연관규칙을 시각화 하기

install.packages('sna')
install.packages('rgl')
library(sna)
library(rgl)

b2 <- t( as.matrix(x)) %*% as.matrix(x)            # 희소 행렬 : 컬럼과 row에 둘다 이름이 들어가도록
b2

# beer bread cola diapers eggs milk
# beer       3     2          1       3           1      2
# bread     2     4          1       3           1      3
# cola        1     1          2       2           0      2
# diapers  3     3          2       4           1      3
# eggs       1     1          0       1           1      0
# milk        2     3          2       3           0      4


# diag( b2 ) : 대각선 행렬에 대한 결과
# diag( diag(b2) ) : b2 의 대각선 행렬만 나오고 나머지는 0 으로 출력

b3 <- b2 - diag ( diag(b2) )        # 대각선 행렬만 0 으로 나오고 나머지 행렬은 자기값 그대로 출력
b3

gplot(b3 , displaylabel=T , vertex.cex=sqrt(diag(b2)) , vertex.col = "green" , edge.col="blue" ,
      boxed.labels=F , arrowhead.cex = .3 , label.pos = 3 , edge.lwd = b3*2) 

# ※ 설명 : displaylabel = T 는 품목명 출력하는 옵션
# vertex.cex 는 출력되는 동그라미의 크기
# vertex.col 는 동그라미 색
# edge.col 는 연결선 색
# boxed.labels 품목명에 box를 둘러주는 옵션
# arrowhead.cex 는 화살표의 크기를 나타내는 옵션
# label.pos 는 품목명의 위치 ( 1은 원 아래, 2는 원 왼쪽,, )
# edge.lwd 는 연결선의 굵기를 나타내는 옵션
