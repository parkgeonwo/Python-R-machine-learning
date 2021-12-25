# 스팸메일 분류 모델의 카파 통계량을 구하시오 !
  
sms_result <- read.csv("sms_results.csv")
  
install.packages("vcd")
library(vcd)
  
table( sms_result$actual_type, sms_result$predict_type )
  
# ham spam
# ham  1203    4
# spam   31  152
  
Kappa( table( sms_result$actual_type, sms_result$predict_type ) )
  
# value      ASE        z       Pr(>|z|)
# Unweighted 0.8825 0.01949 45.27        0
# Weighted   0.8825 0.01949 45.27        0
  
# 0.8 이상이므로 매우좋은 일치이다.
  
  