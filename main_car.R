rm(list = ls())
gc()

setwd("~/R/ML/CAR")

# sum(is.na(testdata$CATEGORY_KIND_CNT))

#####################################################
traindata = readRDS('traindata.rds')
testdata = readRDS('testdata.rds')

traindata = as.data.frame(traindata)
testdata = as.data.frame(testdata)

traindata = traindata[!is.na(traindata$MOTOR_FLG),]
testdata = testdata[!is.na(testdata$MOTOR_FLG),]


traindata = traindata[traindata$MOTOR_FLG=='0',]
testdata = testdata[testdata$MOTOR_FLG=='0',]
traindata$MOTOR_FLG = NULL
testdata$MOTOR_FLG = NULL

traindata = traindata[traindata$IS_COMPLEX=='1',]
testdata = testdata[testdata$IS_COMPLEX=='1',]
traindata$IS_COMPLEX = NULL
testdata$IS_COMPLEX = NULL

setwd("~/R/ML/COMPLEX")

#####################################################
traindata$START_DATETIME = NULL
traindata$INSURE_COMPLEX=NULL
traindata$INSURE_BURGLAR=NULL
traindata$INSURE_LIAB=NULL
traindata$INSURE_PASSENGER=NULL
traindata$INSURE_OTHERS=NULL
traindata$INSURE_FORCE=NULL
traindata$INSURE_FORCE_INJURE=NULL
traindata$INSURE=NULL

tmp =  ifelse(traindata$LOSS_RATE_COMPLEX>0,1,0)
tmp[is.na(tmp)]=0
traindata$LOSS_RATE_COMPLEX=tmp

tmp =  ifelse(traindata$LOSS_RATE_BURGLAR>0,1,0)
tmp[is.na(tmp)]=0
traindata$LOSS_RATE_BURGLAR=tmp

tmp =  ifelse(traindata$LOSS_RATE_LIAB>0,1,0)
tmp[is.na(tmp)]=0
traindata$LOSS_RATE_LIAB=tmp

tmp =  ifelse(traindata$LOSS_RATE_PASSENGER>0,1,0)
tmp[is.na(tmp)]=0
traindata$LOSS_RATE_PASSENGER=tmp

tmp =  ifelse(traindata$LOSS_RATE_OTHERS>0,1,0)
tmp[is.na(tmp)]=0
traindata$LOSS_RATE_OTHERS=tmp

tmp =  ifelse(traindata$LOSS_RATE>0,1,0)
tmp[is.na(tmp)]=0
traindata$LOSS_RATE=tmp

testdata$START_DATETIME = NULL
testdata$INSURE_COMPLEX=NULL
testdata$INSURE_BURGLAR=NULL
testdata$INSURE_LIAB=NULL
testdata$INSURE_PASSENGER=NULL
testdata$INSURE_OTHERS=NULL
testdata$INSURE_FORCE=NULL
testdata$INSURE_FORCE_INJURE=NULL
testdata$INSURE=NULL

tmp =  ifelse(testdata$LOSS_RATE_COMPLEX>0,1,0)
tmp[is.na(tmp)]=0
testdata$LOSS_RATE_COMPLEX=tmp

tmp =  ifelse(testdata$LOSS_RATE_BURGLAR>0,1,0)
tmp[is.na(tmp)]=0
testdata$LOSS_RATE_BURGLAR=tmp

tmp =  ifelse(testdata$LOSS_RATE_LIAB>0,1,0)
tmp[is.na(tmp)]=0
testdata$LOSS_RATE_LIAB=tmp

tmp =  ifelse(testdata$LOSS_RATE_PASSENGER>0,1,0)
tmp[is.na(tmp)]=0
testdata$LOSS_RATE_PASSENGER=tmp

tmp =  ifelse(testdata$LOSS_RATE_OTHERS>0,1,0)
tmp[is.na(tmp)]=0
testdata$LOSS_RATE_OTHERS=tmp

tmp =  ifelse(testdata$LOSS_RATE>0,1,0)
tmp[is.na(tmp)]=0
testdata$LOSS_RATE=tmp

gc()

#####################################################
# traindata$LOSS_RATE_COMPLEX=NULL
traindata$LOSS_RATE_BURGLAR=NULL
traindata$LOSS_RATE_LIAB=NULL
traindata$LOSS_RATE_PASSENGER=NULL
traindata$LOSS_RATE_OTHERS=NULL
traindata$LOSS_RATE=NULL
# testdata$LOSS_RATE_COMPLEX=NULL
testdata$LOSS_RATE_BURGLAR=NULL
testdata$LOSS_RATE_LIAB=NULL
testdata$LOSS_RATE_PASSENGER=NULL
testdata$LOSS_RATE_OTHERS=NULL
testdata$LOSS_RATE=NULL

#####################################################

library(caret)
library(xgboost)

TargetColumn = 'LOSS_RATE_COMPLEX'
ID_column = 'CONTRACT_NO'

Train_Y = traindata[[TargetColumn]]
Train_X = traindata[,-which(colnames(traindata) %in% c(TargetColumn,ID_column))]
Test_Y = testdata[[TargetColumn]]
Test_X = testdata[,-which(colnames(testdata) %in% c(TargetColumn,ID_column))]

nrounds = 4
set.seed(1001)
folds = createFolds(factor(Train_Y), k = 4, list = FALSE)

dev_result <-  rep(0, nrow(Train_X)) 
pred_te <- rep(0, nrow(Test_X))


for (this.round in 1:nrounds){      
  valid <- c(1:length(Train_Y)) [folds == this.round]
  dev <- c(1:length(Train_Y)) [folds != this.round]
  
  dtrain<- xgb.DMatrix(data= as.matrix(Train_X[dev,]), 
                       label= Train_Y[dev],missing=NA)
  
  dvalid <- xgb.DMatrix(data= as.matrix(Train_X[valid,]) , 
                        label= Train_Y[valid],missing=NA)
  
  param = list(objective = "binary:logistic", 
               eval_metric = "auc",
               max_depth = 10,
               eta = 0.05,
               gamma = 5,
               subsample = 0.7,   
               colsample_bytree = 0.7,
               min_child_weight = 50,  
               colsample_bylevel = 0.7,
               lambda = 1, 
               alpha = 0,
               booster = "gbtree",
               silent = 0,
               scale_pos_weight = 6
  ) 
  model<- xgb.train(data = dtrain,
                    params= param, 
                    nrounds = 50, 
                    verbose = T, 
                    list(val1=dtrain , val2 = dvalid) ,   
                    early_stopping_rounds = 50 , 
                    print_every_n = 1,
                    maximize = T
  )
  
  saveRDS(model,paste0('model_car_',this.round,'.rds'))
  pred = predict(model, xgb.DMatrix(data= as.matrix(Train_X[valid,]) ,missing=NA)  )
  dev_result[valid] = pred  
  pred_test  = predict(model,  xgb.DMatrix(data= as.matrix(Test_X) ,missing=NA))
  pred_te = pred_te +pred_test
}


library(pROC)
pred_test = pred_te/nrounds
auc(Test_Y,pred_test)

threshold <- function(predict, response) {
  r <- pROC::roc(response, predict)
  r$thresholds[which.max(r$sensitivities + r$specificities)]
}

# 找出最佳閥值，如果測試資料沒有答案的話就沒有 test_threshold
train_threshold = threshold(dev_result,Train_Y)
test_threshold = threshold(pred_test,Test_Y)
cat("valid threshold：",train_threshold,"; testdata threshold：",test_threshold)

Pred_Y = ifelse(pred_test > 0.50,'1','0')
# confusionMatrix(Pred_Y,Test_Y)

confusion_matrix = table(Pred_Y,Test_Y,dnn = c("預測", "實際"))
confusion_matrix
accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix)
LH=diag(confusion_matrix)/colSums(confusion_matrix)
accuracy
LH


#####################
df = data.frame(CONTRACT_NO = testdata$CONTRACT_NO,PRED_Y= Pred_Y)
setwd("~/R/ML/CAR")
df2 = read_csv('df.csv')

df_tmp = merge(df,df2,by='CONTRACT_NO', all.x = TRUE)

table(df_tmp$PRED_Y)
sum(as.numeric(df_tmp$PREMIUM_COMPLEX[df_tmp$PRED_Y==1]))
sum(as.numeric(df_tmp$PREMIUM_COMPLEX[df_tmp$PRED_Y==0]))

sum(as.numeric(df_tmp$INSURE_COMPLEX[df_tmp$PRED_Y==1]),na.rm=T)
sum(as.numeric(df_tmp$INSURE_COMPLEX[df_tmp$PRED_Y==0]),na.rm=T)





