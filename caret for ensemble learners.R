#continuing with the food/nutrition data set on orange juice
#we continue our tutorial using the caret package

#install.packages(c('caret', 'skimr', 'RANN', 'randomForest', 'fastAdaboost', 'gbm', 'xgboost', 'caretEnsemble', 'C50', 'earth'))
#The final value used for the model was mtry = 6.

#evaluate performance of multiple machine learning algorithms
#adaboost
set.seed(100)

# Train the model using adaboost
model_adaboost = train(Purchase ~ ., data=trainData, method='adaboost', 
                       tuneLength=2, trControl = fitControl)
model_adaboost

#final values used for the model were nIter = 50 and method = Adaboost.M1.

#set.seed(100)
#Training xgBoost Dart (extreme boosting)
# Train the model using MARS
model_xgbDART = train(Purchase ~ ., data=trainData, method='xgbDART', tuneLength=5,
                trControl = fitControl, verbose=F)
model_xgbDART

#gamma' was held constant at a value of 0
#he final values used for the model were nrounds = 200, max_depth = 2, eta
#= 0.3, gamma = 0, subsample = 1, colsample_bytree = 0.6, rate_drop =
#  0.5, skip_drop = 0.05 and min_child_weight = 1.

#SVM
set.seed(100)

# Train the model using MARS
model_svmRadial = train(Purchase ~ ., data=trainData, method='svmRadial', 
                        tuneLength=15, trControl = fitControl)
model_svmRadial

#'sigma' was held constant at a value of 0.06414448

#use the resamples function to compare models
# Compare model performances using resample()
models_compare <- resamples(list(ADABOOST=model_adaboost, RF=model_rf, 
                                 XGBDART=model_xgbDART, MARS=model_mars3, SVM=model_svmRadial))

#Summary of the models performances
summary(models_compare)


#Draw box plots to compare models
#review ROC, Specificity and Sensitivity
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)

#must now Ensemble the predictions
library(caretEnsemble)

# Stacking Algorithms - Run multiple algos in one call.
trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

algorithmList <- c('rf', 'adaboost', 'earth', 'xgbDART', 'svmRadial')

set.seed(100)
models <- caretList(Purchase ~ ., data=trainData, trControl=trainControl, methodList=algorithmList) 
results <- resamples(models)
summary(results)

library(caretEnsemble)

# Stacking Algorithms - Run multiple algos in one call.
trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

algorithmList <- c('rf', 'adaboost', 'earth', 'xgbDART', 'svmRadial')

set.seed(100)
models <- caretList(Purchase ~ ., data=trainData, trControl=trainControl, methodList=algorithmList) 
results <- resamples(models)
summary(results)

#Plot the resamples output to compare the models.
# Box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

#predictions of multiple models to form a final prediction
#possible to combine these predicted values from multiple models somehow and 
#make a new ensemble that predicts better

#use function caretStack BUT do not use the same trainControl you used to build the models.
# Create the trainControl
set.seed(101)
stackControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

# Ensemble the predictions of `models` to form a new combined prediction based on glm
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)

# Predict on testData
stack_predicteds <- predict(stack.glm, newdata=testData4)
head(stack_predicteds)

