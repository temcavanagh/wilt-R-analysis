---
  title: "R Notebook"
output: html_notebook
---
  
  
  ```{r}
getwd()
setwd("/Users/temueracavanagh/Documents/Documents/Data Science/STAT6020 - Predictive Analytics/Project")

wilt = read.csv("wilt_training.csv", stringsAsFactors = TRUE)
wilt_test = read.csv("wilt_testing.csv", stringsAsFactors = TRUE)
```

Data pre-procesing & splitting
```{r}
# Remove row 480 from test 
wilt_test = wilt_test[-c(480),]

# Merge data sets
wilt = rbind(wilt, wilt_test)

# Check merged data
wilt[!complete.cases(wilt),] # No missing values
dim(wilt) # Check dimensions
nlevels(wilt$class) # Check class factor levels
table(wilt$class) # Check classifications
str(wilt)
table(wilt_test$class)
str(wilt_test)

# Split train/test sets
set.seed(0) # Set seed for reproducibility 

# Select 80% of data as sample from total 'n' rows of the data  
sample = sample.int(n = nrow(wilt), size = floor(.8*nrow(wilt)), replace = F)
wilt_train = wilt[sample, ]
wilt_test = wilt[-sample, ]
```


Rescale variables? Lab wk 10 pg 3 



Summary plots

```{r}
library(ggplot2)

ggplot(wilt, aes(y=Mean_Green, x=Mean_Red)) +
  geom_point(aes(colour=class))

ggplot(wilt, aes(y=SD_pan, x=Mean_Red)) +
  geom_point(aes(colour=class))

#install.packages("plotly")
library(plotly)

pairs(wilt[,2:6], col=wilt$class)

```

```{r}
qplot(Mean_Green, Mean_Red, data=wilt, color=class)
```

```{r}
#install.packages("GGally")
library(GGally)

ggpairs(wilt_test, columns = 2:6, ggplot2::aes(colour=class)) 

```



Decision trees #############################

```{r}
library(tree)

tree.train = tree(formula = class~., data = wilt_train)

plot(tree.train)
text(tree.train)

tree.train
summary(tree.train)

# Predict on test set
tree.predict = predict(tree.train, newdata = wilt_test, type = "class")
summary(tree.predict)
table(tree.predict, wilt_test$class)

(900+44)/968
# 97.52066% accuracy on test set
44/(44+5)
# 89.79592 accuracy of w on test set


# Pruning
tree.train.prune = prune.tree(tree.train, best=6, method="misclass")
summary(tree.train.prune)

tree.train.prune2 = prune.tree(tree.train, best=4, method="deviance")
summary(tree.train.prune2)

plot(tree.train.prune)
text(tree.train.prune)

plot(tree.train.prune2)
text(tree.train.prune2)

predict.tree.prune = predict(tree.train.prune, wilt_test, type = "class")
table(predict.tree.prune, wilt_test$class)
(915+34)/968
# 98.03719%
34/(34+15)
# 69.38776%

predict.tree.prune2 = predict(tree.train.prune2, wilt_test, type = "class")
table(predict.tree.prune2, wilt_test$class)
(906+35)/968
# 97.21074%
35/(35+14)
# 71.42857%

```


text 8.3

```{r}
tree.class = tree(class~., data = wilt_train)

summary(tree.class)
plot(tree.class)
text(tree.class, pretty = 0)
tree.class

# Test
set.seed(0)

tree.predict = predict(tree.class, wilt_test, type = "class")
summary(tree.predict)
table(tree.predict, wilt_test$class)

(900+44)/968
# 97.52066%
44/(44+5)
# 89.79592% on w

# Pruning

set.seed(0)
cv.classifier = cv.tree(tree.class, FUN=prune.misclass)
names(cv.classifier)
cv.classifier
# Best size = 13 ################### CHECK
# Now plot
par(mfrow=c(1,2))
plot(cv.classifier$size, cv.classifier$dev, type = "b")
plot(cv.classifier$k, cv.classifier$dev, type = "b")
# apply pruning
prune.classifier = prune.misclass(tree.class, best = 15) ##### CHECK best
plot(prune.classifier)
text(prune.classifier,pretty=0)

# Predict on test data set
prune.predict = predict(prune.classifier, wilt_test, type = "class")
table(prune.predict, wilt_test$class)

# New test set
(915+38)/968
# 98.45041% accuracy on test
38/(38+11)
# 77.55102% accuracy of w on test set 
```



Tree visualisation #1

```{r}
library(visNetwork)
library(sparkline)
library(rpart)

class.rp = rpart(class ~ ., data = wilt_train)
class.rp
visTree(class.rp)
summary(class.rp)


rp.matrix = confusionMatrix(rp.predict, wilt_test$class)
rp.matrix
rp.matrix$overall['Accuracy']
rp.matrix$byClass['Specificity']

# Predict on test set
rp.predict = predict(class.rp, wilt_test, type = "class")
table(rp.predict, wilt_test$class)

(910+42)/968
# 98.34711% accuracy on test set
42/(42+7)
# 85.71429% accuracy of w on test set 


############ Cross validation #############
set.seed(0)
tree.class.cv = printcp(class.rp)
# Matrix
tree.class.cv

# Automatically find index of cost-complexity parameter with min CV error:
min_misclass = which.min(tree.class.cv[, "xerror"]) # Optional, can be done manually
# Automatically resolve ties (taking the simplest model) - Optional, can be done manually: 
while(min_misclass < length(tree.class.cv[ , "xerror"]) &&
      tree.class.cv[min_misclass, "xerror"]==tree.class.cv[min_misclass+1, "xerror"]){ min_misclass = min_misclass + 1
}
# Automatically extract value of cost-complexity parameter corresponding to min CV error: 
CP = tree.class.cv[min_misclass, "CP"] # Optional, can be done manually
# Prune and extract best tree corresponding to min CV error:
tree.class.pruned = prune(tree = class.rp, cp = CP)
tree.class.pruned # Decision-Tree Rules
visTree(tree.class.pruned)
summary(tree.class.pruned)

# Predict using test set
rp.prune.predict = predict(tree.class.pruned, wilt_test, type = "class")
table(rp.prune.predict, wilt_test$class)

(910+40)/968
# 98.1405% accuracy on test set

40/(40+9)
# 81.63265% accuracy of w on test set 
```



####### Random forests ##########################

```{r}
library(randomForest)
library(caret)

set.seed(0)
forest.train = randomForest(class ~ ., data = wilt_train, 
                            mtry = 5, importance = TRUE, ntree = 1000)
forest.train

importance(forest.train)

plot(forest.train, main = "Random forest error rate")

varImpPlot(forest.train, main = "Importance of Variables")


# Predict
forest.test = predict(forest.train, wilt_test, type = "class")
table(forest.test, wilt_test$class)
confusionMatrix(forest.test, wilt_test$class)

# ntree = 2
(909+33)/968
# 97.31405%
33/(33+16)
# 67.34694%

# ntree = 500
(914+44)/968
# 98.96694%
44/(44+5)
# 89.79592%

# ntree = 1000
(914+45)/968
# 99.07025%
45/(45+4)
# 91.83673%


# TUNERF????????

```




SVM  #############################

```{r}
svm_class = svm(class~., data=wilt_train)
svm_class
summary(svm_class)

plot(svm_class, data = wilt_train, Mean_Green~Mean_Red, 
     slice = list(GLCM_pan=1, Mean_NIR=1, SD_pan=1), xaxs="i")

# Predict on training set
svm_training_pred = predict(svm_class, wilt_train)
table(svm_training_pred, wilt_train$class)

(3653+142)/3870
# 98.06202%
142/(142+70)
# 66.98113%

# Predict on test
svm_predict = predict(svm_class, wilt_test)
table(svm_predict, wilt_test$class)
(918+35)/968
# 98.45041%
35/(35+14)
# 71.42857%

```


SVM Linear kernel

```{r}
library(e1071)
library(tidyverse)

set.seed(0)
# NOTE: Ignore WARNINGS "reaching max number of iterations" if any: 
tune.out=tune(svm, class~., data=wilt_train, kernel="linear",
              ranges=list(cost=c(0.001,0.01,0.1,1,10,100))) 
summary(tune.out)
bestmod = tune.out$best.model
summary(bestmod)

# Training set
bestmod_train = predict(bestmod, wilt_train)
table(bestmod_train, wilt_train$class)
(3614+133)/3870
# 96.82171%
133/(133+79)
# 62.73585%

# Test set
bestmod_test = predict(bestmod, wilt_test)
table(bestmod_test, wilt_test$class)
# New test set
(910+30)/968
# 97.10744%
30/(30+19)
# 61.22449%
```


SVM Radial kernel

```{r}
set.seed(0)
# NOTE: Ignore WARNINGS "reaching max number of iterations" if any: 
radtune.out=tune(svm, class~., data=wilt_train, kernel='radial',
                 ranges=list(cost=c(0.001,0.01,0.1,1,10,100),
                             gamma=c(0.01,0.1,1,2,3,4)))

summary(radtune.out)
bestmodrad = radtune.out$best.model

bestmodrad

bestmodrad_train = predict(bestmodrad, wilt_train)
table(bestmodrad_train, wilt_train$class)
# Training set
(3640+190)/3870
# 98.96641%
190/(190+22)
# 89.62264%

bestmodrad_test = predict(bestmodrad, wilt_test)
table(bestmodrad_test, wilt_test$class)
# Test set
(912+47)/968
# 99.07025%
47/(47+2)
# 95.91837%    ##################################################

```

ROC 
```{r}
library(ROCR) 

rocplot=function(pred, truth, ...){
  predob = prediction (pred, truth)
  perf = performance (predob , "tpr", "fpr") 
  plot(perf ,...)}

fitted = attributes(predict(bestmodrad, wilt_train, decision.values=TRUE))$decision.values

par(mfrow=c(1,2))
rocplot(fitted, wilt_train$class, main="Training Data")

fitted = attributes(predict(bestmodrad, wilt_train, decision.values=TRUE))$decision.values
rocplot(fitted, wilt_train$class, main="Test Data") # SVM TRAIN
fitted = attributes(predict(bestmodrad, wilt_test, decision.values=TRUE))$decision.values
rocplot(fitted, wilt_test$class, add=T,col="red") # SVM TEST
fitted = attributes(predict(tree.class.pruned, wilt_test, decision.values=TRUE))$decision.values
rocplot(fitted, wilt_test$class, add=T,col="blue") # 

```


```{r}
results = cbind(rp.matrix$overall['Accuracy'], 
                rp.matrix$byClass['Specificity'],
                prune.matrix$overall['Accuracy'],
                prune.matrix$byClass['Specificity'],
                forest.matrix$overall['Accuracy'],
                forest.matrix$byClass['Specificity'],
                svm.test.matrix$overall['Accuracy'],
                svm.test.matrix$byClass['Specificity'],
                svm.linear.test.matrix$overall['Accuracy'],
                svm.linear.test.matrix$byClass['Specificity'],
                svm.radial.test.matrix$overall['Accuracy'],
                svm.radial.test.matrix$byClass['Specificity'])
results

labs = data.frame(Method=c("Decision Tree",
                           "Pruned Decision Tree",
                           "Random Forest",
                           "Support Vector Machine (OOB)",
                           "Linear Support Vector Machine",
                           "Radial Support Vector Machine"))

accuracy = data.frame(Accuracy=c(rp.matrix$overall['Accuracy'],
                                 prune.matrix$overall['Accuracy'],
                                 forest.matrix$overall['Accuracy'],
                                 svm.test.matrix$overall['Accuracy'],
                                 svm.linear.test.matrix$overall['Accuracy'],
                                 svm.radial.test.matrix$overall['Accuracy']))

specificity = data.frame(Specificity=c(rp.matrix$byClass['Specificity'],
                                       prune.matrix$byClass['Specificity'],
                                       forest.matrix$byClass['Specificity'],
                                       svm.test.matrix$byClass['Specificity'],
                                       svm.linear.test.matrix$byClass['Specificity'],
                                       svm.radial.test.matrix$byClass['Specificity']))

results_table = cbind(labs, accuracy, specificity)
results_table

```

```{r}
library(tinytex)
```


