#EXAMPLE 4 
#CARET PACKAGE

#install.packages(c('caret', 'skimr', 'RANN', 'randomForest', 'C50', 'earth'))
# Load the caret package
library(caret)
  # Import dataset
#C:\\Users\\redbe\\OneDrive\\Documents\\Data\\orangejuice.csv
orange <- read.csv(file.choose(), header=TRUE)

# Structure of the dataframe
str(orange)

# See top 6 rows and 10 columns
head(orange[, 1:10])

## Create the training and test datasets
set.seed(100)

#1: Get row numbers for the training data
trainRowNumbers <- createDataPartition(orange$Purchase, p=0.8, list=FALSE)

#2: Create the training  dataset
trainData <- orange[trainRowNumbers,]

#3: Create the test dataset
testData <- orange[-trainRowNumbers,]

# Store X and Y for later use.
x = trainData[, 2:18]
y = trainData$Purchase #character variable

#createDataPartition() takes as input the Y variable in the source dataset and the percentage data that should go into training as the p argument. It returns the rownumbers that should form the training dataset.

#set list=F, to prevent returning the result as a list.

#skimr package provides a nice solution to show key descriptive stats for each column.
#skimr::skim_to_wide() produces a dataframe containing the descriptive stats of each of the columns. 
#The dataframe output includes a histogram drawn without any plotting help.

library(skimr)
skimmed <- skim(trainData)
skimmed[, c(1:5, 9:11, 13, 15:16)]

#Notice the number of missing values for each feature, mean, median, proportion 
#split of categories in the factor variables, percentiles and the histogram in the last column.

#impute missing values using preProcess()
#feature is a continuous variable, it is a common practice to replace the missing values with the mean of the column. 
#feature is a categorical variable, replace the missings with the most frequently occurring value, the mode.

#INSTEAD
#predict the missing values by considering the rest of the available variables as predictors. 
#Do imputation is the k-Nearest Neighbors. 
#To predict the missing values with k-Nearest Neighbors using preProcess():

#set the method=knnImpute for k-Nearest Neighbors and apply it on the training data. This creates a preprocess model.
#Then use predict() on the created preprocess model by setting the newdata argument on the same training data.
#Use bagImpute as an alternative imputation algorithm.

# Create the knn imputation model on the training data
preProcess_missingdata_model <- preProcess(trainData, method='knnImpute')
preProcess_missingdata_model

#shown are various preprocessing steps done in the process of knn imputation.
#has centered (subtract by mean) variables, 
#ignored , 
#used k= (considered x nearest neighbors) to predict the missing values 
#scaled (divide by standard deviation) 16 variables.

# Use the imputation model to predict the values of missing data points
library(RANN)  # required for knnInpute
trainData <- predict(preProcess_missingdata_model, newdata = trainData)
anyNA(trainData)

#used for centering and scaling. method = "knnImpute" enforces centering and scaling
#the data because the kNN algo makes little sense without those processes.
#use method = "bagImpute" or method = "medianImpute" if you don't want the observations to be transformed.

#create One-Hot Encoding (dummy variables)
#categorical column as one of the features, it needs to be converted to numeric in order for it to be used by the machine learning algorithms.

# ensure the dummyVars model is built on the training data alone and that model is in turn used to create the dummy vars on the test data.

#One-Hot Encoding
#Creating dummy variables is converting a categorical variable to as many binary variables as here are categories.
dummies_model <- dummyVars(Purchase ~ ., data=trainData)

#Create the dummy variables using predict. The Y variable (Purchase) will not be present in trainData_mat.
trainData_mat <- predict(dummies_model, newdata = trainData)

#Convert to dataframe
trainData <- data.frame(trainData_mat)

#See the structure of the new dataset
str(trainData)

#preprocess to transform the data - 
#range: Normalize values so it ranges between 0 and 1
#center: Subtract Mean
#scale: Divide by standard deviation
#BoxCox: Remove skewness leading to normality. Values must be > 0
#YeoJohnson: Like BoxCox, but works for negative values.
#expoTrans: Exponential transformation, works for negative values.
#pca: Replace with principal components
#ica: Replace with independent components
#spatialSign: Project the data to a unit circle

#method=range in preProcess()
preProcess_range_model <- preProcess(trainData, method="range") #normalize
trainData <- predict(preProcess_range_model, newdata = trainData)

# Append the Y variable
trainData$Purchase <- y
apply(trainData[, 1:10], 2, FUN=function(x){c('min'=min(x), 'max'=max(x))})

#visualize the importance of variables using featurePlot() - produces lattice graphs
#how to gauge if a given X is an important predictor of Y

#Move Purchase to the front if wanted
#trainData = trainData[,c(ncol(trainData),1:(ncol(trainData)-1))]

trainData$Purchase<- as.factor(trainData$Purchase)
#Feature Plotting and Selection

#mean and the placement of the two boxes are seen to be different.
featurePlot(x = trainData[, 1:18], 
            y = trainData$Purchase, 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

#using density plots
#For a variable to be important, you would expect the density curves to be significantly 
#different for the 2 classes, both in terms of the height (kurtosis) and placement (skewness).
featurePlot(x = trainData[, 1:18], 
            y = trainData$Purchase, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

#only says which variables are likely to be important to predict Y. 

#feature selection using recursive feature elimination (rfe)
#determine what features are important to predict the Y

#RFE works in 3 broad steps:
#1: Build a ML model on a training dataset and estimate the feature importances on the test dataset.
#2: Keeping priority to the most important variables, iterate through by building
#models of given subset sizes, that is, subgroups of most important predictors 
#determined from step 1. Ranking of the predictors is recalculated in each iteration.
#3: The model performances are compared across different subset sizes to arrive 
#at the optimal number and list of final predictors.

set.seed(100)
#options(warn=-1) #suppress warnings in the global settings

subsets <- c(1:5, 10, 15, 18)

#flexibility to control what algorithm rfe uses and how it cross validates by defining the rfeControl()
ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

lmProfile <- rfe(x=trainData[, 1:18], trainData$Purchase,
                 sizes = subsets,
                 rfeControl = ctrl) #cross validate
lmProfile

#RFE also takes two important parameters, sizes and rfeControl.
#sizes determines what all model sizes (the number of most important features) the rfe should consider.
#rfeControl parameter on the other hand receives the output of the rfeControl() as values.
#If you look at the call to rfeControl() we set what type of algorithm and what 
#cross validation method should be used.
#repeatedcv which implements k-Fold cross validation repeated 5 times, 

#Once rfe() is run, the output shows the accuracy and kappa (and sd) for different model sizes.
#The final selected model subset size is marked with a * in the right selected column.

#Training and Tuning the model

#To know what models caret supports
modelnames <- paste(names(getModelInfo()), collapse=', ')
modelnames

modelLookup("rf")

#Multivariate Adaptive Regression Splines (MARS) model by setting the method='earth'.
modelLookup('earth')

# Set the seed for reproducibility
set.seed(100)

# Train the model using MARS and predict on the training data itself.
model_mars = train(Purchase ~ ., data=trainData, method='earth')
fitted <- predict(model_mars)

#train() does the following functions:
#Cross validates the model
#Tunes the hyper parameters for optimal model performance
#Chooses the optimal model based on a given evaluation metric
#Preprocesses the predictors (i.e. using preProcess())

model_mars
plot(model_mars, main="Model Accuracies using MARS")

#How to compute variable importance?
varimp_mars <- varImp(model_mars)
plot(varimp_mars, main="Variable Importance using MARS")

#Prepare the test dataset and predict
#1: Impute missing values 
testData2 <- predict(preProcess_missingdata_model, testData)  

#2: Create one-hot encodings (dummy variables)
testData3 <- predict(dummies_model, testData2)

# Step 3: Transform the features to range between 0 and 1
testData4 <- predict(preProcess_range_model, testData3)

# View
head(testData4[, 1:10])

# Predict on testData
predicted <- predict(model_mars, testData4)
head(predicted)

#Confusion Matrix
#compare predictions (data) vs actuals (reference)
#mode='everything'= most classification evaluation metrics are computed.
#data` and `reference` should be factors with the same levels
testData$Purchase = as.factor(testData$Purchase)

# Compute the confusion matrix
confusionMatrix(reference = testData$Purchase, data = predicted, 
                mode="everything", 
                positive="MM")

#hyperparameter tuning to optimize the model for better performance
#two main ways to do hyper parameter tuning using the train():
# - Set the tuneLength - corresponds to the number of unique values for the tuning parameters caret will consider while forming the hyper parameter combinations.
# - Define and set the tuneGrid - explicitly control what values should be considered for each parameter,

#Cross validation method can be one amongst:
#'boot': Bootstrap sampling
#'boot632': Bootstrap sampling with 63.2% bias correction applied
#'optimism_boot': The optimism bootstrap estimator
#'boot_all': All boot methods.
#'cv': k-Fold cross validation
#'repeatedcv': Repeated k-Fold cross validation
#'oob': Out of Bag cross validation
#'LOOCV': Leave one out cross validation
#'LGOCV': Leave group out cross validation

#summaryFunction can be twoClassSummary if Y is binary class or
#multiClassSummary if the Y has more than 2 categories.
#set classProbs=T for the probability scores instead of directly predicting
#the class based on a predetermined cutoff of 0.5.

# Define (setup) the training control
fitControl <- trainControl( method = 'cv', # k-fold cross validation
  number = 5,                      # number of folds
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = T,                  # should class probabilities be returned
  summaryFunction=twoClassSummary  # results summary function
)

#1: Tune hyper parameters by setting tuneLength
set.seed(100)
model_mars2 = train(Purchase ~ ., data=trainData, method='earth', tuneLength = 5, metric='ROC', trControl = fitControl)
model_mars2

#2: Predict on testData and Compute the confusion matrix
predicted2 <- predict(model_mars2, testData4)
confusionMatrix(reference = testData$Purchase, data = predicted2, mode='everything', positive='MM')

#Tuning parameter 'degree' was held constant at a value of 1
#ROC was used to select the optimal model using the largest value.
#You can set the tuneGrid instead of tuneLength.

#1: Define the tuneGrid
marsGrid <-  expand.grid(nprune = c(2, 4, 6, 8, 10), 
                         degree = c(1, 2, 3))

#2: Tune hyper parameters by setting tuneGrid
set.seed(100)
model_mars3 = train(Purchase ~ ., data=trainData, method='earth', metric='ROC', tuneGrid = marsGrid, trControl = fitControl)
model_mars3

#3: Predict on testData and Compute the confusion matrix
predicted3 <- predict(model_mars3, testData4)
confusionMatrix(reference = testData$Purchase, data = predicted3, 
    mode="everything", 
    positive="MM")

#Train the model using rf
set.seed(100)
model_rf = train(Purchase ~ ., data=trainData, method='rf', tuneLength=5, trControl = fitControl)
model_rf

#we will continue here on our next class evaluating performance of 
#multiple machine learning algorithms

