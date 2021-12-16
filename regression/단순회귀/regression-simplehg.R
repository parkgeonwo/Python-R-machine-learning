
# 1. 데이터를 로드합니다.
hg <- read.csv("simple_hg.csv", header = T)
head(hg)


# 2. 데이터를 산포도 그래프로 시각화합니다.
plot(hg$input ~ hg$cost, data = hg, pch = 21, col = 'blue', bg = 'blue')



# 3. 회귀분석을 해서 회귀계수인 기울기와 절편을 구합니다.

model <- lm(hg$input ~ hg$cost, data = hg)
model

#
# Coefficients:
#   (Intercept)      hg$cost  
# 62.929        2.186 

# 4. 2번에서 시각화한 산포도 그래프에 회귀직선을 겹처서 그립니다.

abline(model, col = 'red')


# 5. 그래프 제목을 회귀직선의 방정식으로 출력되게 합니다.

title( paste( '매출 =',model$coefficients[2], 'x  광고비 + ', model$coefficients[1]  ) )




# 6. 오차 그리는 코드

y_hat <- predict( model, cost = hg$cost )       #  input 매출액 예측값 출력 ( 직선 그래프의 값 )
y_hat                                                                    # 15개의 예측값 출력

join <- function(i){                      # join 이라는 이름의 함수 생성
  lines( c( hg$cost[i], hg$cost[i]), c( hg$input[i], y_hat[i] ), col = 'green' )     # 녹색라인그래프
}

sapply(1:19, join)                # 1:19들을 join 함수에 적용하는 것 , 판다스의 apply나 map같은 느낌?




# ※ 오차와 잔차와의 차이 ?
  
# 1. 오차 : 모집단에서 실제값이 회귀선과 비교했을때의 차이 ( 실제값과 예측값과의 차이 )
# 2. 잔차 : 표본에서 실제값과 회귀선과 비교했을때의 차이 ( 실제값과 예측값과의 차이 )
