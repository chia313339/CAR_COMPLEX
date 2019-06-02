rm(list = ls())
gc()

# 全在server環境開發

# 設定自己工作空間 需修改
setwd("~/R/ML/CAR")

# 讀取server上資料
dataset = readRDS('~/../public/LIAB/dataset_all_preprocess_20190517.rds')
dataset = as.data.frame(dataset)

# 移除不要的欄位，保留 CONTRACT_NO 當辨識欄位，START_DATETIME 切測試資料用
drop_list =c('APPLLTE_NO',  'PRODUCT_CODE', 'PRODUCT_POLICY_CODE',	'PRODUCT_SUB_CODE',	'POLICY_NO', 'POLICY_STATUS',	'POLICY_RESULT_CODE',	'APPLLTE_DATE', 'ISSUE_DATE',	'END_DATETIME',	'INSRNCE_TARGET', 'CONTINUE_POLICY_NO',	'AGENT_ID',	'COUNSEL_ID', 'CUSTOMER_CLASSFY',	'CUSTOMER_ID',	'IS_MAIN_INSURED', 'BIRTHDAY',	'OCCU_CATEGORY_CODE', 'OCCU_CATEGORY_CODE_IND',	'POLICY_TYPE',	'CAR_NO', 'ENGINE_NO',	'ORIGIN_ISSUE_DATE',	'MADE_DATE', 'LEGAL_PREMIUM',	'TRADE_FORCE_NO',	'TRADE_START_DATE', 'ACCEPT_OFFICER',	'VEHICLE_KIND_NAME',	'MODEL_NAME', 'IS_WHITE_LIST')
drop_ind =  names(dataset)  %in% drop_list
dataset = dataset[!drop_ind]


# 轉換函數
label_encoder = function(vec){
  levels = sort(unique(vec))
  function(x){
    match(x, levels)
  }
}

# 有類別的變數全轉 label
LE_UNIT_NO = label_encoder(dataset$UNIT_NO)
tmp_col = as.integer(LE_UNIT_NO(dataset$UNIT_NO))
dataset$UNIT_NO = tmp_col

LE_BUSINESS_ORIGIN = label_encoder(dataset$BUSINESS_ORIGIN)
tmp_col = as.integer(LE_BUSINESS_ORIGIN(dataset$BUSINESS_ORIGIN))
dataset$BUSINESS_ORIGIN = tmp_col

LE_COPERATE_COMPANY_NO = label_encoder(dataset$COPERATE_COMPANY_NO)
tmp_col = as.integer(LE_COPERATE_COMPANY_NO(dataset$COPERATE_COMPANY_NO))
dataset$COPERATE_COMPANY_NO = tmp_col

LE_AGENT_DIV_NO = label_encoder(dataset$AGENT_DIV_NO)
tmp_col = as.integer(LE_AGENT_DIV_NO(dataset$AGENT_DIV_NO))
dataset$AGENT_DIV_NO = tmp_col

LE_COUNSEL_DIV_NO = label_encoder(dataset$COUNSEL_DIV_NO)
tmp_col = as.integer(LE_COUNSEL_DIV_NO(dataset$COUNSEL_DIV_NO))
dataset$COUNSEL_DIV_NO = tmp_col

LE_BROKER_NO = label_encoder(dataset$BROKER_NO)
tmp_col = as.integer(LE_BROKER_NO(dataset$BROKER_NO))
dataset$BROKER_NO = tmp_col

LE_AGENT_KIND = label_encoder(dataset$AGENT_KIND)
tmp_col = as.integer(LE_AGENT_KIND(dataset$AGENT_KIND))
dataset$AGENT_KIND = tmp_col

tmp_col = as.integer(ifelse(dataset$SEX=='M',1,ifelse(dataset$SEX=='F',0,NA)))
dataset$SEX = tmp_col

LE_BUSINESS_TYPE = label_encoder(dataset$BUSINESS_TYPE)
tmp_col = as.integer(LE_BUSINESS_TYPE(dataset$BUSINESS_TYPE))
dataset$BUSINESS_TYPE = tmp_col

LE_OCCU_TYPE1_CODE = label_encoder(dataset$OCCU_TYPE1_CODE)
tmp_col = as.integer(LE_OCCU_TYPE1_CODE(dataset$OCCU_TYPE1_CODE))
dataset$OCCU_TYPE1_CODE = tmp_col

LE_MODEL_FULL_NO = label_encoder(dataset$MODEL_FULL_NO)
tmp_col = as.integer(LE_MODEL_FULL_NO(dataset$MODEL_FULL_NO))
dataset$MODEL_FULL_NO = tmp_col

LE_VEHICLE_KIND_NO = label_encoder(dataset$VEHICLE_KIND_NO)
tmp_col = as.integer(LE_VEHICLE_KIND_NO(dataset$VEHICLE_KIND_NO))
dataset$VEHICLE_KIND_NO = tmp_col

LE_KIND_NO = label_encoder(dataset$KIND_NO)
tmp_col = as.integer(LE_KIND_NO(dataset$KIND_NO))
dataset$KIND_NO = tmp_col

LE_BODY_NO = label_encoder(dataset$BODY_NO)
tmp_col = as.integer(LE_BODY_NO(dataset$BODY_NO))
dataset$BODY_NO = tmp_col

LE_MODEL_SUB_NAME = label_encoder(dataset$MODEL_SUB_NAME)
tmp_col = as.integer(LE_MODEL_SUB_NAME(dataset$MODEL_SUB_NAME))
dataset$MODEL_SUB_NAME = tmp_col

LE_MODEL_NO = label_encoder(dataset$MODEL_NO)
tmp_col = as.integer(LE_MODEL_NO(dataset$MODEL_NO))
dataset$MODEL_NO = tmp_col

LE_RISK_LEVEL = label_encoder(dataset$RISK_LEVEL)
tmp_col = as.integer(LE_RISK_LEVEL(dataset$RISK_LEVEL))
dataset$RISK_LEVEL = tmp_col

LE_SUSPECT_LEVEL = label_encoder(dataset$SUSPECT_LEVEL)
tmp_col = as.integer(LE_SUSPECT_LEVEL(dataset$SUSPECT_LEVEL))
dataset$SUSPECT_LEVEL = tmp_col

tmp_col = as.integer(ifelse(is.na(dataset$COMPUTER_CALCULAT_CODE),0,1))
dataset$COMPUTER_CALCULAT_CODE = tmp_col

tmp_col = as.integer(ifelse(dataset$IS_CONTINUE_POLICY_NO=='N',0,1))
dataset$IS_CONTINUE_POLICY_NO = tmp_col

LE_MAXIMUM_LOAD_UNIT = label_encoder(dataset$MAXIMUM_LOAD_UNIT)
tmp_col = as.integer(LE_MAXIMUM_LOAD_UNIT(dataset$MAXIMUM_LOAD_UNIT))
dataset$MAXIMUM_LOAD_UNIT = tmp_col

dataset$AUTO_HAND = as.integer(dataset$AUTO_HAND)

tmp_col = as.integer(ifelse(dataset$IS_NEW_CAR=='N',0,1))
dataset$IS_NEW_CAR = tmp_col

tmp_col = as.integer(ifelse(dataset$QUESTION_CAR_NO_IND=='N',0,1))
dataset$QUESTION_CAR_NO_IND = tmp_col

tmp_col = as.integer(ifelse(dataset$QUESTION_ENGIN_IND=='N',0,1))
dataset$QUESTION_ENGIN_IND = tmp_col

tmp_col = as.integer(ifelse(dataset$IS_NOT_FORCE=='N',0,1))
dataset$IS_NOT_FORCE = tmp_col

tmp_col = as.integer(ifelse(dataset$MOTOR_FLG=='N',0,1))
dataset$MOTOR_FLG = tmp_col

# FORCE_INJURE_IND_L3Y 不知為何R判定為文字  轉一下
dataset$FORCE_INJURE_IND_L3Y = as.integer(dataset$FORCE_INJURE_IND_L3Y)


# 移除異常資料
dataset = dataset[dataset$ORIGIN_ISSUE_DUR>=0,]
dataset = dataset[dataset$AGE>=0,]

# 將 START_DATETIME 轉 Date 並做切割
dataset$START_DATETIME = as.Date(dataset$START_DATETIME)

testdata = dataset[dataset$START_DATETIME>='2017-01-01',]
traindata = dataset[dataset$START_DATETIME<'2017-01-01',]

# 存 RDS 備份
saveRDS(traindata,'traindata.rds')
saveRDS(testdata,'testdata.rds')

# 釋放一下空間
rm(dataset)
gc()

#####################################################
#####################################################

# 將 START_DATETIME 還有理賠金額欄位刪除
# 將所有損率轉為 0/1 類別

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
#####################################################

# 因只要做整單風險，將其他損率類別刪除

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
#####################################################
library(caret)
library(xgboost)
library(pROC)

# 標示識別欄位
TargetColumn = 'LOSS_RATE_COMPLEX'
ID_column = 'CONTRACT_NO'

# 定義資料集
Train_Y = traindata[[TargetColumn]]
Train_X = traindata[,-which(colnames(traindata) %in% c(TargetColumn,ID_column))]
rm(traindata)
gc()
Test_Y = testdata[[TargetColumn]]
Test_X = testdata[,-which(colnames(testdata) %in% c(TargetColumn,ID_column))]
rm(testdata)
gc()

# 建立fold數
nrounds = 4
set.seed(100)
folds = createFolds(factor(Train_Y), k = nrounds, list = FALSE)

# 放結果的空向量
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
               max_depth = 20,
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
               scale_pos_weight = 0.142
               # tree_method = 'gpu_hist'
  ) 
  model<- xgb.train(data = dtrain,
                    params= param, 
                    nrounds = 5000, 
                    verbose = T, 
                    list(val1=dtrain , val2 = dvalid) ,   
                    early_stopping_rounds = 50 , 
                    print_every_n = 1,
                    maximize = T
  )
  
  # 儲存這次訓練的模型
  saveRDS(model,paste0('model_',this.round,'.rds'))
  pred = predict(model, xgb.DMatrix(data= as.matrix(Train_X[valid,]) ,missing=NA)  )
  dev_result[valid] = pred  
  pred_test  = predict(model,  xgb.DMatrix(data= as.matrix(Test_X) ,missing=NA))
  pred_te = pred_te +pred_test
}

# 將 4 個模型預測結果平均，計算AUC
pred_test = pred_te/nrounds
auc(Test_Y,pred_test)

# 計算閥值函數
threshold <- function(predict, response) {
  r <- pROC::roc(response, predict)
  r$thresholds[which.max(r$sensitivities + r$specificities)]
}

# 找出最佳閥值
train_threshold = threshold(dev_result,Train_Y)
test_threshold = threshold(pred_test,Test_Y)
cat("valid threshold：",train_threshold,"; testdata threshold：",test_threshold)

# 將找出來的閥值切割預測結果
Pred_Y = ifelse(pred_test > 0.0145,'1','0')
# confusionMatrix(Pred_Y,Test_Y)

# 建立混淆矩陣並顯示結果
confusion_matrix = table(Pred_Y,Test_Y,dnn = c("預測", "實際"))
confusion_matrix
accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix)
LH=diag(confusion_matrix)/colSums(confusion_matrix)
accuracy
LH

# threshold = 0.475
# 
# accuracy
# [1] 0.7246446
# 
# > LH
# 0         1 
# 0.7246418 0.7247065 

# 計算主成分
importance_matrix <- xgb.importance(colnames(Train_X), model = model)
xgb.plot.importance(importance_matrix, rel_to_first = TRUE, xlab = "Relative importance",top =10)


