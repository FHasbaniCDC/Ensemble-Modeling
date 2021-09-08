#Classification

derm <- read.csv(file.choose())
#Preparing the data
#Classification with gbm
#Classification with caret train method
#install.packages("gbm")
library(gbm)
library(caret)

derm$class <- as.factor(derm$class)
indexes = createDataPartition(derm$class, p = .75, list = F)
train = derm[indexes, ]
test = derm[-indexes, ]

#EDA

#Classification with gbm
mod_gbm = gbm(class ~.,
              data = train,
              distribution = "multinomial",
              cv.folds = 10, #10 fold cross validation
              shrinkage = .01,
              n.minobsinnode = 10,
              n.trees = 200)
print(mod_gbm)

#predict test data
pred = predict.gbm(object = mod_gbm,
                   newdata = test,
                   n.trees = 200,
                   type = "response"
                   )
#?predict.gbm

#obtaining class names with the highest prediction value.
labels = apply(pred,1, which.max)
result = data.frame(test$class, as.factor(labels))
print(result)

str(test)
#check confusion matrix
cm = confusionMatrix(as.factor(test$class), as.factor(labels))
print(cm)

#Classification with caret train method
#use train() function for model fit
tc = trainControl(method = "repeatedcv", number = 10)

#train model
model = train(class ~., data=train, method="gbm", trControl=tc)
print(model)

#predict
pred = predict(model, test)
result = data.frame(test$class, pred)
print(result)

#check confusion matrix
cm = confusionMatrix(test$class, as.factor(pred))
print(cm) 

#accuracy increased from 96.59 to 96.59
#changes happened in the classification from classes 2 and 4
