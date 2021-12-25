# . 스팸분류기 모델의 민감도와 특이도를 R 로 구하시오 !
  

sms_result <- read.csv("sms_results.csv" , stringsAsFactors = T)

install.packages("caret")
library(caret)

sensitivity( sms_result$predict_type, sms_result$actual_type, positive = 'spam' )           # [1] 0.8306011

specificity( sms_result$predict_type, sms_result$actual_type, negative = 'ham' )          # [1] 0.996686
