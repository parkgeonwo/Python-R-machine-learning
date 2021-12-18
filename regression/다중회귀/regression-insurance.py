# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 17:52:52 2021

@author: parkk
"""

# ▩ 미국민 의료비 데이터를 파이썬으로 회귀모델 생성하기

# 데이터셋 : insurance.csv
# 종속변수 : expenses
# 독립변수 : age, sex, bmi, children, smoker, region

# 1. 데이터를 로드합니다.

insur = pd.read_csv("c:\\data\\insurance.csv")

# 2. 결측치를 확인합니다.

print( insur.isnull().sum() )

# 3. 종속변수의 정규성을 확인합니다.

insur.expenses.plot( kind = 'hist' )



# 4. 회귀모델을 생성합니다.

import statsmodels.formula.api as smf

model = smf.ols( formula = 'expenses ~ age + sex + bmi + children + smoker + region', data = insur )

# 5. 모델을 훈련시킵니다.

result = model.fit()

# 6. 분석결과를 확인합니다.

print( result.summary() )



print(result.params)          # 기울기 쪽만 출력

# Intercept                      -11941.562461
# sex[T.male]                  -131.352014        ---------> 남성은 여성에 비해 매년 의료비가 131 달러 적게 들거라 예상
# smoker[T.yes]              23847.476695    ---------> 흡연자는 비흡연자보다 매년 의료비가 23,860 달러 더 많이 든다. 
# region[T.northwest]     -352.790096     -----------> 지역별로는 북동지역이 북서,남동,남서에 비해 의료비가 더든다.
# region[T.southeast]     -1035.595701
# region[T.southwest]     -959.305829
# age                                  256.839171   ---------> 나이가 1년 증가때마다 평균 의료비 256 더든다. 
# bmi                                  339.289863   ----------> 비만지수 1 증가시 의료비 339 더든다
# children                            475.688916  ----------> 부양가족이 1명 늘때마다 475달러 더든


# 문제. 비만인 사람은 의료비가 더 지출이 되는지 bmi30 이라는 파생변수를 추가하고
# 		   다시 모델을 만들어서 결정계수가 올라가는지 확인하시오 !
		
insur = pd.read_csv("c:\\data\\insurance.csv")

insur['bmi30'] = (insur['bmi'] >= 30).astype(int)

####### 함수만들어서 파생변수 생성 ###
def func_1(x):
    if x >= 30:
        return 1
    else:
        return 0

insur['bmi30'] = insur['bmi'].apply(func_1)

#############################################

import statsmodels.formula.api as smf

model = smf.ols( formula = 'expenses ~ age + sex + bmi + children + smoker + region + bmi30', data = insur )

result = model.fit()

print( result.summary() )      # 0.751 ---------> 0.756으로 올라갔습니다.


# 문제. 비만이면서 흡연까지 하게되면 의료비가 더 증가하는지
# 		  bmi30_yes 파생변수를 추가해서 결정계수가 더 올라가는지 확인하시오 !

insur = pd.read_csv("c:\\data\\insurance.csv")

insur['bmi30'] = (insur['bmi'] >= 30).astype(int)
insur['bmi30_yes'] = ( (insur['bmi'] >= 30) & ( insur['smoker'] == 'yes' ) ).astype(int)

import statsmodels.formula.api as smf

model = smf.ols( formula = 'expenses ~ age + sex + bmi + children + smoker + region + bmi30 + bmi30_yes', data = insur )

result = model.fit()

print( result.summary() )      #  0.756  ---------> 0.864 으로 올라갔습니다.
