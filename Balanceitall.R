#Function for ROC curve and AUC score returns the auc score
auc_score<-function(prediction,original){

  predrocr<-ROCR::prediction(prediction,original)
  perfrocr<-ROCR::performance(predrocr,"tpr","fpr")
  p<-plot(perfrocr)
  auc<-performance(predrocr,"auc")@y.values
  return(auc)
}
#First model Logistic regression
logisticReg<-function(data,dependent.name){
  print("##########-----------Starting process for Logistic regression------------###############")
  set.seed(3)
  index<-createDataPartition(data[[ncol(data)]],p=0.7,list=FALSE)
  data[dependent.name]<-ifelse(data[[ncol(data)]]==0,"N","Y")#converting dependent variable to N/Y
  data[[dependent.name]]<-as.factor(data[[dependent.name]])#converting to factor class
  names(data)[ncol(data)]<-paste("Dependent")#For Formula in the model the dependent varible name is changed to a common name
  data.train<-data[index,] #train
  data.test<-data[-index,] #test
  tc<-trainControl(method = "cv",number = 10,classProbs = T,summaryFunction = twoClassSummary,verboseIter = TRUE)
  model.logreg<-train(Dependent~.,data.train,method="glm",trControl=tc,metric="ROC")
  logreg.pred<-predict(model.logreg,newdata = data.test[-ncol(data.test)],type="prob")
  auc_logreg<-auc_score(logreg.pred$Y,data.test[ncol(data.test)])#Through AUC function
  title("logistic regression")#for the ROC plot
  #Calibration of predicted values through platt scaling
  data.cali.lgrg<-data.frame(logreg.pred,data.test[ncol(data.test)])
  colnames(data.cali.lgrg)<-c("x","y")
  model.cali<-glm(y~x,data.cali.lgrg,family = binomial)
  pred.cali.lgrg=predict(model.cali,data.cali.lgrg[-2],type="response")
  print("logistic regression done")
  return(list("model"=model.logreg,"Predicted Values"=logreg.pred$Y,"Calibrated Values"=pred.cali.lgrg,"AUC"=auc_logreg))
}
#Second model re-sampling through SMOTE and Ranger on balanced data (Ratio is set by the user)
SamplingRF<-function(data,dependent.name){
  print("##########-------SMOTE+Random Forest--------############")
  # library(caret)
  set.seed(3)
  index<-createDataPartition(data[[ncol(data)]],p=0.7,list=FALSE)
  data[dependent.name]<-ifelse(data[[ncol(data)]]==0,"N","Y")
  data[[dependent.name]]<-as.factor(data[[dependent.name]])
  names(data)[ncol(data)]<-paste("Dependent")
  data.train<-data[index,]
  data.test<-data[-index,]


  n<-ncol(data.train)
  output<-as.factor(data.train[,ncol(data.train)])
  input<-data.train[,-ncol(data.train)]

  # boxplot(ladies_ind$AVG  _DISTINCT_CATEG_BOUGHT)
  print(table(data.train[ncol(data.train)]))#Values in each label of the dependent variable
  a<-readline("What value to set for percOver? Depends on the ratio you want to maintain(for 50-50% put 100) ")
  b<-readline("What value to set for percUnder? Depends on the ratio you want to maintain(for 50-50% put 200) ")
  bald<-unbalanced::ubBalance(X= input, Y=output, type="ubSMOTE",positive = "Y", percOver=as.numeric(a), percUnder=as.numeric(b), verbose=TRUE)
  balancedData<-cbind(bald$X,bald$Y)
  table(balancedData$`bald$Y`)#now train on this data
  nrow(balancedData)
  #creating random forest(ranger for faster implementation) model on this
  # detach("package:unbalanced", unload=TRUE)
  # detach("package:mlr", unload=TRUE)
  ctrl <- trainControl(method = "cv", number = 10,classProbs = TRUE,summaryFunction = twoClassSummary,verboseIter = TRUE)
  model.rf <- train(`bald$Y` ~ ., data = balancedData, method = "ranger",
                    trControl = ctrl,metric="ROC"#,tuneGrid=expand.grid("mtry"=c(),"min.node.size"=1,"splitrule"="gini")
  )#tuning the hyperparameters can be done here
  pred_rf <- predict(model.rf, newdata=data.test[,-ncol(data.test)],type="prob")
  auc_samplingrf<-auc_score(pred_rf$Y,data.test$Dependent)#through AUC function
  title("sampling+rf")#For ROC curve
  #CAlibration of predicted values
  data.cali.rf<-data.frame(pred_rf$Y,data.test[ncol(data.test)])
  colnames(data.cali.rf)<-c("x","y")
  model.cali<-glm(y~x,data.cali.rf,family = binomial)
  pred.cali.rf=predict(model.cali,data.cali.rf[-2],type="response")
  print("Sampling + Rf done")
  return(list("model"=model.rf,"Predicted Values"=pred_rf$Y,"Calibrated Values"=pred.cali.rf,"AUC"=auc_samplingrf))
}
#Third model XGboost
Xgboost<-function(data,dependent.name){
  print("###########------------Starting process for xgboost-------------###############")
  #checking for factor variable as xgboost boost doesn't take factor variables
  #converting them to dummy variables
  count=0
  for(i in 1:ncol(data)){
    if(class(data[,i])=="factor"){
      count= count + 1
    }
  }
  if(count>=1){
    print("one hot encoding required for application of xgboost, running it")
    # library(dummies)
    dummy.var <- dummyVars(~ ., data = data, fullRank = TRUE)
    dummy_data<-predict(dummy.var,data)
    dummy_data<-as.data.frame(dummy_data)
  }else{
    dummy_data<-data
  }
  dependent.name.new<-names(dummy_data[ncol(dummy_data)])#name of dependent variable may be changed
  # library(xgboost)
  set.seed(3)
  index<-createDataPartition(data[[ncol(data)]],p=0.7,list=FALSE)
  # Full data set
  data_variables <- as.matrix(dummy_data[,-ncol(dummy_data)])
  data_label <- dummy_data[,dependent.name.new]
  data_matrix <- xgboost::xgb.DMatrix(data = as.matrix(dummy_data), label = data_label)
  # split train data and make xgb.DMatrix
  train_data   <- data_variables[index,]
  train_label  <- data_label[index]
  train_matrix <- xgboost::xgb.DMatrix(data = train_data, label = train_label)
  # split test data and make xgb.DMatrix
  test_data  <- data_variables[-index,]
  test_label <- data_label[-index]
  test_matrix <- xgboost::xgb.DMatrix(data = test_data, label = test_label)

  xgb_params <- list("objective" = "binary:logistic",
                     "eval_metric" = "auc")#tuning parameters can be put in this

  nround    <- 50 # number of XGBoost rounds
  cv.nfold  <- 10
  # Fit cv.nfold * cv.nround XGB models and save OOF predictions
  cv_model <- xgboost::xgb.cv(params = xgb_params,
                     data = train_matrix,
                     nrounds = nround,
                     nfold = cv.nfold,
                     verbose = TRUE,
                     prediction = TRUE)

  model.xgb <- xgboost::xgb.train(params = xgb_params,
                         data = train_matrix,
                         nrounds = nround)
  # Predict hold-out test set
  pred_xgb <- predict(model.xgb, newdata = test_matrix)
  auc_xgboost=auc_score(pred_xgb,test_label)#Through AUC function
  title("xgboost")#for ROC curve
  #Calibration of predicted variables
  data.cali.xgb<-data.frame(pred_xgb,test_label)
  colnames(data.cali.xgb)<-c("x","y")
  model.cali<-glm(y~x,data.cali.xgb,family = binomial)
  pred.cali.xgb<-predict(model.cali,data.cali.xgb[-2],type="response")
  print("xgboost done")
  return(list("model"=model.xgb,"Predicted Values"=pred_xgb,"Calibrated Values"=pred.cali.xgb,"AUC"=auc_xgboost))
}
#Fourth model SVM
svm<-function(data,dependent.name){
  print("##########-----------Starting process for SVM------------###############")
  set.seed(3)
  index<-createDataPartition(data[[ncol(data)]],p=0.7,list=FALSE)
  data[dependent.name]<-ifelse(data[[ncol(data)]]==0,"N","Y")
  data[[dependent.name]]<-as.factor(data[[dependent.name]])
  names(data)[ncol(data)]<-paste("Dependent")
  data.train<-data[index,]
  data.test<-data[-index,]
  # return(data.train)
  tc<-trainControl(method = "cv",number = 10,classProbs = T,summaryFunction = twoClassSummary,verboseIter = TRUE)
  model.svm<-train(Dependent~.,data.train,method="svmRadialWeights",preProc = c("center","scale"),trControl=tc,tuneLength=3,metric="ROC")
  pred_svm<-predict(model.svm,newdata = data.test[-ncol(data.test)],type="prob")
  auc_svm<-auc_score(pred_svm$Y,data.test[ncol(data.test)])
  title("SVM")
  data.cali.svm<-data.frame(pred_svm,data.test[ncol(data.test)])
  colnames(data.cali.svm)<-c("x","y")
  model.cali<-glm(y~x,data.cali.svm,family = binomial)
  pred.cali.svm<-predict(model.cali,data.cali.svm[-2],type="response")
  print("SVM done")
  return(list("model"=model.svm,"Predicted Values"=pred_svm$Y,"Calibrated Values"=pred.cali.svm,"AUC"=auc_svm))
}
#Fifth model RUSBoost which is basically random under sampling with random forest applied through Adaboost
RUSBoost<-function(data,dependent.name){
  # library(ebmc)
  # library(caret)
  print("##########-----------Starting process for RUSBoost------------###############")
  set.seed(3)
  index<-createDataPartition(data[[ncol(data)]],p=0.7,list=FALSE)
  data[[dependent.name]]<-as.factor(data[[dependent.name]])
  names(data)[ncol(data)]<-paste("Dependent")
  data.train<-data[index,]
  data.test<-data[-index,]
  ratio<-table(data.train$Dependent)[1]/table(data.train$Dependent)[2]#ratio of majority label and minority label is required
  ratio<-ratio[1][[1]]
  model.rus<-rus(Dependent~.,data.train,size=10,alg="rf",rf.ntree=100,ir=ratio)
  rus.pred<-predict(model.rus,newdata = data.test[-ncol(data.test)],type="prob")
  auc_rus<-auc_score(rus.pred,data.test[ncol(data.test)])#Thorugh AUC function
  title("RUSBoost")#for ROC curve
  #Calibration of the predicted variables
  data.cali.rus<-data.frame(rus.pred,data.test[ncol(data.test)])
  colnames(data.cali.rus)<-c("x","y")
  model.cali<-glm(y~x,data.cali.rus,family = binomial)
  pred.cali.rus<-predict(model.cali,data.cali.rus[-2],type="response")
  print("RUSBoost done")
  return(list("model"=model.rus,"Predicted Values"=rus.pred,"Calibrated Values"=pred.cali.rus,"AUC"=auc_rus))
}






#' Solves the problem of classification on imbalanced datasets
#'
#'
#' \code{ShubhamR} Main function for the package, enter the pre-processed dataset and the name of the dependent variable in quotes.Make sure that the dependent variable is in the end and labels are 1,0 and is numeric. When asked for which model to use, Enter the model name as after the semicolon
#'
#' For class imbalanced classification this package can be used. This package consists of Five models which help to achieve that.
#' The function when run will ask for which model to run and will also ask for the proportion to maintain when SMOTE+RF is used(recommended 100:1000 will generate 16.67% proportion).
#' It will return the model function,predicted values,calibrated values, AUC score and a dataframe Modelperf(in the Global environment).ModelPerf consists of different model performance measures and will be appended each time the function is run
#'
#' @param dataset Pre-proccessed dataset with last coloumn as dependent with 0,1 labels with numeric class
#' @param dependent.name dependent variable name in "" quotes
#'
#' @export
ShubhamR<-
  function(data,dependent.name){


  #this is required for the Model performance calculation as original test labels are required for validation
  set.seed(3)
  index<-createDataPartition(data[[ncol(data)]],p=0.7,list=FALSE)
  data.test<-data[-index,]

  if(dependent.name!=names(data[ncol(data)])){
    return(print("Dependent Variable name doesn't match with last Feature name in the dataset, exit the function and write the correct name"))
  }

  #input from the user for the model
  model<-readline("Available models are:(enter the name after : )
                  1)Logistic Regression :logisticReg
                  2)Sampling(SMOTE)&Ranger  :SamplingRF
                  3)Xgboost  :Xgboost
                  4)SVM  :SVM
                  5)RUSBoost :RUSBoost
                  Which ever is required type the name after : (only one) ")
  #switch function
  values<-switch(model,
                 logisticReg=logisticReg(data,dependent.name),
                 SamplingRF=SamplingRF(data,dependent.name),
                 Xgboost=Xgboost(data,dependent.name),
                 SVM=svm(data,dependent.name),
                 RUSBoost=RUSBoost(data,dependent.name))
  ###----evaluation of best model by different measures-----########
  # library(dplyr)
  #created data set with predictions, demi decile labels and original values
  eval.data<-data.frame(values$`Predicted Values`,ntile(values$`Predicted Values`,20),data.test[ncol(data.test)])
  colnames(eval.data)<-c("pred","ntile","original")
  eval.data.5perc<-eval.data %>% mutate(label=ifelse(ntile==20,1,0))
  eval.data.10perc<-eval.data %>% mutate(label=ifelse(ntile==20|ntile==19,1,0))
  eval.data.15perc<-eval.data %>% mutate(label=ifelse(ntile==20|ntile==19|ntile==18,1,0))
  #this is with the proportion of labels in the train set
  trainprop<-(min(prop.table(table(data.test[ncol(data.test)]))))*nrow(data.test)
  eval.data.trainprop<- eval.data %>% mutate(ntile=ntile(values$`Predicted Values`,trainprop),label=ifelse(ntile==trainprop,1,0))
  # Confusion Matrices
  c.5perc<-confusionMatrix(as.factor(eval.data.5perc$label),as.factor(eval.data$original))
  c.10perc<-confusionMatrix(as.factor(eval.data.10perc$label),as.factor(eval.data$original))
  c.15perc<-confusionMatrix(as.factor(eval.data.15perc$label),as.factor(eval.data$original))
  c.trainprop<-confusionMatrix(as.factor(eval.data.trainprop$label),as.factor(eval.data$original))
  #dataframe with Model performance wrt each cutoff
  # Going out in the environment and will apend each time the package is run, has to be initialized with different datasets
  if(exists("ModelPerf")==FALSE){
    ModelPerf<<-data.frame(matrix(data = 0, nrow = 1,ncol = 13))
  }
  names(ModelPerf) = c("algorithm", "cutoff",names(c.5perc$byClass))
  newtemp1 = c(model, "Top 5% cutoff", c.5perc$byClass)
  newtemp2 = c(model, "Top 10% cutoff", c.10perc$byClass)
  newtemp3 = c(model, "Top 15% cutoff", c.15perc$byClass)
  newtemp4 = c(model, "Train proportion as cutoff", c.trainprop$byClass)
  ModelPerf<<-rbind(ModelPerf,newtemp1,newtemp2,newtemp3,newtemp4)
  return(values)
}
