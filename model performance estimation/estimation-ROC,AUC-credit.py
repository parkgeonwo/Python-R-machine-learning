# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 18:03:52 2021

@author: parkk
"""

#▩ 판다스로 독일 은행의 채무 불이행자를 예측하는 기계학습 모델 만들기


import  pandas  as  pd 
credit =  pd.read_csv("c:\\data\\credit.csv")

credit2 = pd.get_dummies(credit.iloc[ :  , :-1] )

x =  credit2.to_numpy()                #   학습 시킬 데이터 생성 
y =  credit.iloc[ :  , -1].to_numpy()   #   정답 데이터 생성

from  sklearn.model_selection  import  train_test_split

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.1, random_state=1)

from  sklearn.preprocessing import  MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)

x_train2 = scaler.transform(x_train)

x_test2 = scaler.transform(x_test)

from  sklearn.tree  import  DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='entropy', max_depth=5)

model.fit( x_train2,  y_train)

result = model.predict( x_test2 )
print( result )

preds2 = model.predict_proba( x_test2 )[ : , 1 ]
print(preds2)

print ( sum( result == y_test ) / len(y_test) )

from  sklearn.metrics  import  confusion_matrix

a = confusion_matrix( y_test, result )
print( a )

result2 = pd.DataFrame( y_test )
result2[result2 == 'yes'] = 1
result2[result2 == 'no'] = 0
result2.columns = ['defaulter']           
y_test2 = list( result2['defaulter'] )                # 불이행자(yes) = 1 / 이행자(no) = 0


############ 그래프 그리기 #############


# 설명 :  metrics.roc_curve( 예측값, 예측확률 )
						     #↑
		# 예측값을 0과 1로 제공해줘야합니다. 1이 관심범주입니다.
		# 예측확률은 관심범주쪽의 확률로 제공해줘야 합니다.

from  sklearn import metrics

fpr, tpr, threshold = metrics.roc_curve( y_test2, preds2 )
roc_auc = metrics.auc(fpr, tpr)
print ( roc_auc )               # 0.7180952380952381

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# method II: ggplot
# from ggplot import *
# df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
# ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')








