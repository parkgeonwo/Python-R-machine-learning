# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 23:21:40 2021

@author: parkk
"""

# sms_results.csv 의 스팸메일 분류 모형의 카파통계량을 파이썬으로 구하시오 !

import pandas as pd
sms = pd.read_csv("c:\\data\\sms_results.csv")

from sklearn.metrics import cohen_kappa_score

print( cohen_kappa_score ( sms['actual_type'] , sms['predict_type']  ) )          # 0.8825202721955789



