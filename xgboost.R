
#xgboost takes several types of input data:
#Dense Matrix: R's dense matrix, i.e. matrix 
#Sparse Matrix: R's sparse matrix, i.e. Matrix::dgCMatrix 
#Data File: local data files
#xgb.DMatrix: its own class is recommended.

#Parameters used in Xgboost
#three types of parameters: 
#1 - General Parameters
#2 - Booster Parameters 
#3 - Task Parameters.

#General parameters refers to which booster we are using to do boosting. 
#common are tree or linear model - 
#silent : The default value is 0. You need to specify 0 for printing running messages, 1 for silent mode.
#booster : The default value is gbtree. You need to specify the booster to use: gbtree (tree based) or gblinear (linear function).
#num_pbuffer : set automatically by xgboost. see xgboost documentation
#num_feature : set automatically by xgboost. see xgboost documentation

#Booster parameters depends on which booster you have chosen. The tree specific parameters -
#eta : default value=0.3. You need to specify step size shrinkage used in update to prevents overfitting. 
#After each boosting step, we can directly get the weights of new features. and eta
#actually shrinks the feature weights to make the boosting process more conservative. The range is 0 to 1. Low eta value means model is more robust to overfitting.
#gamma : default value= 0. You need to specify minimum loss reduction required to make a further partition on a leaf node of the tree. The larger, the more conservative the algorithm will be. The range is 0 to ???. Larger the gamma more conservative the algorithm is.
#max_depth : default value=6. You need to specify the maximum depth of a tree. The range is 1 to ???.
#min_child_weight : default value=1. You need to specify the minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be. The range is 0 to ???.
#max_delta_step: default value=0. Maximum delta step we allow each tree's weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.The range is 0 to ???.
#subsample: default value=1. You need to specify the subsample ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting. The range is 0 to 1.
#colsample_bytree: default value=1. You need to specify the subsample ratio of columns when constructing each tree. The range is 0 to 1.

#Linear Booster Specific Parameters
#lambda and alpha : These are regularization term on weights. Lambda default value assumed is 1 and alpha is 0.
#lambda_bias : L2 regularization term on bias and has a default value of 0.

#Learning Task parameters that decides on the learning scenario
#base_score : The default value is set to 0.5 . You need to specify the initial prediction score of all instances, global bias.
#objective : The default value is set to reg:linear . You need to specify the type of learner you want which includes linear regression, logistic regression, poisson regression etc.
#eval_metric : You need to specify the evaluation metrics for validation data, a default metric will be assigned according to objective( rmse for regression, and error for classification, mean average precision for ranking
#seed : As always here you specify the seed to reproduce the same set of outputs.

#install.packages("xgboost")
library(xgboost)
library(Matrix)
library(data.table)

#install.packages("vcd")
library(vcd)

data(Arthritis)
df <- data.table(Arthritis, keep.rownames = FALSE)
head(df)

str(df)
#2 columns have factor type, one has ordinal type.
#ordered values are: Marked > Some > None

#Grouping per every 10 years
#create groups of age by rounding the real age
#transform to factor so the algorithm will treat age groups as independent values
head(df[,AgeDiscret := as.factor(round(Age/10,0))])
summary(df$AgeDiscret)

#arbitrary split at 40 years old
head(df[,AgeCat:= as.factor(ifelse(Age > 40, "Old", "Young"))])

df[,ID:=NULL]
#list different values for the column Treatment.

levels(df[,Treatment])
#data are stored in a dgCMatrix which is a sparse matrix and label vector is a numeric vector {0,1}
#transform each value of each categorical feature in a binary feature {0, 1}
#observation which has the value Placebo in column Treatment before the transformation will have after the transformation the value 1 in the new column Placebo.
#value 0 in the new column Treated.
sparse_matrix <- sparse.model.matrix(Improved~.-1, data = df)
head(sparse_matrix)

#Improved~.-1 used above means transform all categorical features but column Improved to binary values. 
#The -1 is here to remove the first column 
?sparse.model.matrix

#Create the output numeric vector (this is not as a sparse Matrix)
output_vector = df[,Improved] == "Marked"

#set Y vector to 0
#set Y to 1 for rows where Improved == Marked is TRUE ;
#return Y vector

bst <- xgboost(data = sparse_matrix, label = output_vector, max.depth = 4,
               eta = 1, nthread = 2, nrounds = 10,objective = "binary:logistic")
#Each line shows how well the model explains your data. the lower is better.

#feature importance
importance <- xgb.importance(feature_names = sparse_matrix@Dimnames[[2]], model = bst) #column names of the sparse matrix
head(importance)

#features are classified by Gain
#Gain is the improvement in accuracy brought by a feature to the branches it is on.
#Cover measures the relative quantity of observations concerned by a feature.
#Frequency is a simpler way to measure the Gain. It just counts the number of 
#times a feature is used in all generated trees. use gain before freq.

#count the co-occurrences of a feature and a class of the classification
#use data and label
importanceRaw <- xgb.importance(feature_names = sparse_matrix@Dimnames[[2]], 
                                   model = bst, data = sparse_matrix, label = output_vector)

#Cleaning for better display
importanceClean <- importanceRaw[,`:=`(Cover=NULL, Frequency=NULL)]
head(importanceClean)

xgb.plot.importance(importance_matrix = importanceRaw)

c2 <- chisq.test(df$Age, output_vector)
print(c2)

c2 <- chisq.test(df$AgeDiscret, output_vector)
print(c2)

c2 <- chisq.test(df$AgeCat, output_vector)
print(c2)