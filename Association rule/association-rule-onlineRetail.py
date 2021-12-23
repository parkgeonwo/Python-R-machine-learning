# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 21:08:52 2021

@author: parkk
"""

# 프랑스에서 주문한 상품들의 연관규칙을 출력하시오 !
# 지지도 0.03 이상, 향상도 0.5 이상으로 영국과 똑같이 하세요

df = pd.read_csv("c:\\data\\OnlineRetail.csv", encoding = 'unicode_escape')

df2 =  df.loc[ ~ df.InvoiceNo.str.contains('C'), : ]
df3 = df2.loc[ df2.Country == 'France', : ]

result =  df3.groupby( [ 'InvoiceNo', 'Description' ] )['Quantity'].sum().unstack()

result.fillna(0, inplace = True)

def function(x):
    if x >= 1.0:
        return 1
    else:
        return 0

result2 = result.applymap(function)

from mlxtend.frequent_patterns import apriori

itemsets = apriori( result2, min_support = 0.03, use_colnames = True )
print(itemsets)

from mlxtend.frequent_patterns import association_rules

rules = association_rules( itemsets, metric = 'lift', min_threshold = 0.5 )
print(rules)



