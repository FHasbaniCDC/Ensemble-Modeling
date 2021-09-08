library(purrr)
library(ggplot2)
library(corrplot)
library(tidyverse)
library(xgboost)
library(caret)

power_plant = read.csv(file.choose())
class(power_plant)

#Changing data table to data frame
# Caret faces problems working with tbl, 
# so let's change the data to simple data frame
power_plant = data.frame(power_plant)
message("The class is now ", class(power_plant))
# See first few rows
head(power_plant)

#target = power outage, indepentent - other 4
# Size of  DataFrame
dim(power_plant)

#EDA
#function map from purrr package
map(power_plant, class)
map(power_plant, ~sum(is.na(.)))  

#using ggplot2
power_plant %>% ggplot(aes(AT, PE)) +
  geom_point(color= "blue", alpha = 0.3) +
  ggtitle("Temperature vs Power Output") +
  xlab("Atmospheric Temperature") +
  ylab("Power Output") +
  theme(plot.title = element_text(color="darkred",
                                  size=18,hjust = 0.5),
        axis.text.y = element_text(size=12),
        axis.text.x = element_text(size=12,hjust=.5),
        axis.title.x = element_text(size=14),
        axis.title.y = element_text(size=14))

power_plant %>% ggplot(aes(V, PE)) +
  geom_point(color= "darkgreen", alpha = 0.3) +
  ggtitle("Exhaust Vacuum Speed vs Power Output") +
  xlab("Exhaust Vacuum Speed") +
  ylab("Power Output") +
  theme(plot.title = element_text(color="darkred",size=18,hjust = 0.5),
        axis.text.y = element_text(size=12),
        axis.text.x = element_text(size=12,hjust=.5),
        axis.title.x = element_text(size=14),
        axis.title.y = element_text(size=14))

power_plant %>% ggplot(aes(AP, PE)) +
  geom_point(color= "red", alpha = 0.3) +
  ggtitle("Atmospheric Pressure vs Power Output") +
  xlab("Atmospheric Pressure") +
  ylab("Power Output") +
  theme(plot.title = element_text(color="darkred",size=18,hjust = 0.5),
        axis.text.y = element_text(size=12),
        axis.text.x = element_text(size=12,hjust=.5),
        axis.title.x = element_text(size=14),
        axis.title.y = element_text(size=14))

#Correlation heatmap
correlations = cor(power_plant)
corrplot(correlations, method="color")

#using caret package, perform regression analysis
#partition data
# create training set indices with 80% of data
set.seed(100)  # For reproducibility
# Create index for testing and training data
inTrain <- createDataPartition(y = power_plant$PE, 
           p = 0.8, list = FALSE)
# subset power_plant data to training
training <- power_plant[inTrain,]
# subset the rest to test
testing <- power_plant[-inTrain,]

# Size ratio of training and test dataset
message("As shown below, the training set is about 80%  and the test set is about 20% of the original data")

rbind("Training set" = nrow(training)/nrow(power_plant),
      "Testing set" = nrow(testing)/nrow(power_plant)) %>% 
       round(2)*100

#regression
#Fit linear regression model
#put the predictors on the same scale: mean of zero and unit variance
my_lm = train(training[,1:4], training[,5],
              method = "lm",
              preProc = c("center", "scale"))

message("Linear Regression: Model performance on \n the training set")
my_lm$results[c("RMSE","Rsquared")] %>%
  round(2)
summary(my_lm)

#predict
pred = predict(my_lm, testing[, 1:4])
SSE = sum((testing[,5] -pred)^2)    # sum of squared errors
SST = sum((testing[,5] - mean(training[,5]))^2) # total sum of squares, remember to use training data here

R_square = 1 - SSE/SST
message('R_squared on the test data:')
round(R_square, 2)

SSE = sum((testing[,5] - pred)^2)
RMSE = sqrt(SSE/length(pred))
message("Root mean square error on the test data: ")
round(RMSE, 2)

#see R2 and RMSE
#plot
my_data = as.data.frame(cbind(predicted = pred,
                              observed = testing$PE))

#Plot predictions vs test data
ggplot(my_data,aes(predicted, observed)) +
  geom_point(color = "darkred", alpha = 0.5) + 
  geom_smooth(method=lm)+ ggtitle('Linear Regression ') +
  ggtitle("Linear Regression: Prediction vs Test Data") +
  xlab("Predecited Power Output ") +
  ylab("Observed Power Output") +
  theme(plot.title = element_text(color="darkgreen",size=18,hjust = 0.5),
        axis.text.y = element_text(size=12),
        
        axis.text.x = element_text(size=12,hjust=.5),
        axis.title.x = element_text(size=14),
        axis.title.y = element_text(size=14))

#XGBoost
set.seed(100)  # For reproducibility
# Create index for testing and training data
#inTrain <- createDataPartition(y = power_plant$PE, p = 0.8, list = FALSE)

#Convert the training and testing sets into DMatrixes: 
#DMatrix is the recommended class in xgboost
X_train = xgb.DMatrix(as.matrix(training %>% select(-PE)))
y_train = training$PE

X_test = xgb.DMatrix(as.matrix(testing %>% select(-PE)))
y_test = testing$PE

#Specify cross-validation method and number of folds. Also enable parallel computation
#The simplest parameters are:
#max_depth (maximum depth of the decision trees being trained)
#objective (the loss function being used)
#num_class (the number of classes in the dataset).
#eta 

xgb_trcontrol = trainControl(
  method = "cv",
  number = 5,  
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE)

#specific the hyperparameters to optimize. use the expand grid (see caret random forest example)
#eta parameter gives us a chance to prevent this overfitting
#The eta can be thought of as a learning rate.
#range of 0.1 to 0.3.
xgbGrid <- expand.grid(nrounds = c(100,200),  # this is n_estimators 
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       #below are default parameters
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1)

#train model
set.seed(0) 
xgb_model = train(X_train, y_train,  
        trControl = xgb_trcontrol,
        tuneGrid = xgbGrid,
        method = "xgbTree")

#hyperparameters are chosen
xgb_model$bestTune

#model evaluation
predicted = predict(xgb_model, X_test)
residuals = y_test - predicted
RMSE = sqrt(mean(residuals^2))
cat('The root mean square error of the test data is ', round(RMSE,3),'\n')

#RMSE?

y_test_mean = mean(y_test)

#Calculate total sum of squares
tss =  sum((y_test - y_test_mean)^2 )

#Calculate residual sum of squares
rss =  sum(residuals^2)

#Calculate R-squared
rsq  =  1 - (rss/tss)
cat('The R-square of the test data is ', round(rsq,3), '\n')

#plot actual/predicted
options(repr.plot.width=8, repr.plot.height=4)
my_data = as.data.frame(cbind(predicted = predicted,
                              observed = y_test))

#Plot predictions vs test data
ggplot(my_data,aes(predicted, observed)) + geom_point(color = "darkred", alpha = 0.5) + 
  geom_smooth(method=lm)+ ggtitle('Linear Regression ') + ggtitle("Extreme Gradient Boosting: Prediction vs Test Data") +
  xlab("Predecited Power Output ") + ylab("Observed Power Output") + 
  theme(plot.title = element_text(color="darkgreen",size=16,hjust = 0.5),
        axis.text.y = element_text(size=12), axis.text.x = element_text(size=12,hjust=.5),
        axis.title.x = element_text(size=14), axis.title.y = element_text(size=14))

#gamma parameter can also help with controlling overfitting. 
#gamma specifies the minimum reduction in the loss required to make a further 
#partition on a leaf node of the tree. 
#for example, if creating a new node does not reduce the loss by a certain amount, then it will not create at all.

#booster parameter allows you to set the type of model you will use when building the ensemble.
#The default is gbtree which builds an ensemble of decision trees. 
#If your data is not too complicated, you can go with the faster and simpler gblinear option which builds an ensemble of linear models.
