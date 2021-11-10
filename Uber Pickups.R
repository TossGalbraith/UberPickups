library(caret)
library(regclass)
library(parallel)
library(doParallel)
library(lubridate)


load("UBERDATA.RData") 


UBER <- rbind(TRAIN,ALLDATA)  #stack the 2 dataset together


#Convert the dates
dates <- UBER$pickup_dt
dates_converted <- ymd_hms(dates)

#The year is the same throughout all the data
UBER$Month <- month(dates_converted,label=TRUE)
UBER$Day <- wday(dates_converted,label=TRUE)  
UBER$Hour <- hour(dates_converted)


#After "cleaning", re-make the training and holdout samples
TRAIN.CLEAN <- UBER[1:nrow(TRAIN),]
TRAIN.CLEAN$IDno <- NULL
TRAIN.CLEAN$pickup_dt <- NULL

ALLDATA.CLEAN <- UBER[-(1:nrow(TRAIN)),]



#Estimate generalization errors with 5-fold crossvalidation
fitControl <- trainControl(method="cv",number=5, allowParallel = TRUE) 

#turning on parallelization
cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster) 
#turning off parallelization
# stopCluster(cluster) 
# registerDoSEQ() 


# #Regularized logistic regression
# glmnetGrid <- expand.grid(alpha = seq(0,1,.05),lambda = 10^seq(-4,-1,length=10))
# GLMnet <- train(logpickups~.,data=TRAIN.CLEAN,method='glmnet',tuneGrid=glmnetGrid,
#                                trControl=fitControl, preProc = c("center", "scale"))
# 
# GLMnet  #Look at details of all fits
# GLMnet$bestTune #Gives best parameters
# GLMnet$results #Look at output in more detail
# GLMnet$results[rownames(GLMnet$bestTune),]  #Just the row with the optimal choice of tuning parameter
# #RMSE 0.2196653


# #Random Forest
# treeGrid <- expand.grid(cp=10^seq(-5,-1,length=25))
# TREE <- train(logpickups~.,data=TRAIN.CLEAN,method='rpart', tuneGrid=treeGrid,
#                              trControl=fitControl, preProc = c("center", "scale"))
# 
# TREE  #Look at details of all fits
# plot(TREE) #See how error changes with choices
# TREE$bestTune #Gives best parameters
# TREE$results #Look at output in more detail
# TREE$results[rownames(TREE$bestTune),]  #Just the row with the optimal choice of tuning parameter
# #RMSE 0.1577014


#Boosted Tree
gbmGrid <- expand.grid(n.trees=c(3500),interaction.depth=10,shrinkage=c(.02),n.minobsinnode=c(10))

GBM <- train(logpickups~.,data=TRAIN.CLEAN, method='gbm',tuneGrid=gbmGrid,verbose=FALSE,
             trControl=fitControl, preProc = c("center", "scale"))

GBM  #Look at details of all fits
# plot(GBM) #See how error changes with choices
GBM$bestTune #Gives best parameters
GBM$results #Look at output in more detail
GBM$results[rownames(GBM$bestTune),]  #Just the row with the optimal choice of tuning parameter
#RMSE 0.1329568


# #Neural Network (only 1 hidden layer allowed)
# nnetGrid <- expand.grid(size=1:4,decay=10^( seq(-5,-2,length=10) ) )
# 
# #IMPERATIVE to have trace=FALSE and linout=TRUE for regression; also scaling has to be done of each variable
# NNET <- train(logpickups~.,data=TRAIN.CLEAN,method='nnet',trControl=fitControl,tuneGrid=nnetGrid,
#               trace=FALSE,linout=TRUE,preProc = c("center", "scale"))
# 
# # NNET  #Look at details of all fits
# # plot(NNET) #See how error changes with choices
# NNET$bestTune #Gives best parameters
# NNET$results #Look at output in more detail 
# NNET$results[rownames(NNET$bestTune),]  #Just the row with the optimal choice of tuning parameter
# # RMSE 0.2003256 


# #K-nearest neighbors
# #Tune values for number of neighbors
# knnGrid <- expand.grid(k=1:50)
# KNN <- train(logpickups~.,data=TRAIN.CLEAN, method='knn', trControl=fitControl,tuneGrid=knnGrid,
#                              preProc = c("center", "scale"))
# 
# # KNN  #Look at details of all fits
# # plot(KNN) #See how error changes with choices
# KNN$bestTune #Gives best parameters
# KNN$results #Look at output in more detail
# KNN$results[rownames(KNN$bestTune),]  #Just the row with the optimal choice of tuning parameter
# #RMSE 0.3032231 


# #Polynomial kernel (equivalent to all predictors and all two-way interactions and predictors raised to degree power)
# svmPolyGrid <- expand.grid(degree=2:3, scale=10^seq(-4,-1,by=1), C=2^(2:4) )
# SVMpoly <- train(logpickups~.,data=TRAIN.CLEAN, method='svmPoly', trControl=fitControl,tuneGrid = svmPolyGrid,
#                               preProc = c("center", "scale"))
# 
# SVMpoly  #Look at details of all fits
# plot(SVMpoly) #See how error changes with choices
# SVMpoly$bestTune #Gives best parameters
# SVMpoly$results #Look at output in more detail 
# SVMpoly$results[rownames(SVMpoly$bestTune),]  #Just the row with the optimal choice of tuning parameter
# # Took too long to run > 15 min
