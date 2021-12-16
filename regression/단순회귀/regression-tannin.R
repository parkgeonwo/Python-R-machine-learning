"탄닌 함유량과 애벌래 성장간의 관계에 대한 회귀식을 도출하기"

# 1. 데이터를 로드합니다.
# 2. 데이터를 산포도 그래프로 시각화합니다.
# 3. 회귀분석을 해서 회귀계수인 기울기와 절편을 구합니다.
# 4. 2번에서 시각화한 산포도 그래프에 회귀직선을 겹처서 그립니다.
# 5. 그래프 제목을 회귀직선의 방정식으로 출력되게 합니다.


# 1. 데이터를 로드합니다.
reg <- read.table("regression.txt", header =T)

# 2. 데이터를 산포도 그래프로 시각화합니다.
attach(reg)
plot(growth ~ tannin, data = reg, pch = 21, col = 'blue', bg = 'blue')


# ※ 설명 : plot( y ~ x, data = 데이터프레임명 )

# 3. 회귀분석을 해서 회귀계수인 기울기와 절편을 구합니다.

model <- lm( growth ~ tannin, data = reg )
model

# 
# Coefficients:
#   (Intercept)       tannin  
# 11.756       -1.217  
# ↑                  ↑
# 절편             기울기   


# 4. 2번에서 시각화한 산포도 그래프에 회귀직선을 겹처서 그립니다.

attach(reg)
plot(growth ~ tannin, data = reg, pch = 21, col = 'blue', bg = 'blue')    # 산포도 그래프
model <- lm( growth ~ tannin, data = reg )                                               # 회귀 모델 생성
abline(model, col = 'red' )                                                                           # 회귀모델의 직선의 그래프



# 5. 그래프 제목을 회귀직선의 방정식으로 출력되게 합니다.

model$coefficients[2]            # 기울기
model$coefficients[1]            #  절편

title( paste( '성장률 =',model$coefficients[2], 'x  탄닌 + ', model$coefficients[1]  ) )
