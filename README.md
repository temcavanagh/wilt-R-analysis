# wilt-R-analysis


---
title: A Comparative Assessment of Machine Learning Classification Methods For The
  Detection of Wilt Disease in Pine and Oak Trees
output:
  html_document:
    df_print: paged
  html_notebook: default
  word_document: default
  pdf_document: default
---


by Tem Cavanagh as part of STAT6020 Predictive Analytics (Master of Data Science)

***

<center>

**Abstract:** *This paper examines the classification accuracy of machine learning methods in*
*detecting wilt disease present in Japanese pine and oak trees. A comparative*
*assessment of statistical learning methods is conducted on the wilt dataset*
*whereby the classification accuracy of decision trees, random forest ensembles and *
*support vector machines are analysed. Formal quantitative assessment of results*
*are carried out in combination with qualitative assessment through visualisation* 
*of results. The resulting findings identified a support vector machine with a*
*radial kernel as the most accurate and specific method for classification.* 
*It is hoped that the findings of this project demonstrate the applicability of* 
*statistical learning techniques as preventative measures against forest degradation.*


</center>

***
### Introduction:

Forests are becoming increasingly susceptible to damage and degradation by 
way of plant diseases caused by insects and climate change (Food and Agriculture 
Organization of the United Nations Regional Office for Asia and the Pacific 2010;
Garrett et. al. 2006). 

Discolouration of foliage is often an indicator of a diseased tree (Johnson). 
Johnson et. al. have conducted high resolution satellite image surveys with the
purpose of identifying diseased pine and oak trees in Japan. The resulting images
are shown below:

![Figure 1: Diseased classified trees (Source: Johnson)](/Users/temueracavanagh/Documents/Documents/Data Science/STAT6020 - Predictive Analytics/Project/wilt_image.png){width=50%}

The objective of this project is to assess and compare the classification accuracy of 
statistical learning methods in detecting and classifying diseased trees. The overarching
aim of this project is to demonstrate the applicability of statistical learning techniques as 
preventative measures against forest degradation.

***
### Data:

The data for this project was sourced from the UCI Machine Learning Repository (http://archive.ics.uci.edu/ml/datasets/wilt).

The data was originally collected by the Centre for Environmental Remote Sensing
through satellite imagery which was in turn 'pansharpened' to obtain spatially 
and spectrally accurate image measurements.

The total dimensions of the dataset is 4838 observations across 6 variables.

#### Variable descriptions:  

Variable       | Type   | Description
---------------|--------|----------------------
**class**      | Factor | 'w' (wilt diseased trees), 'n' (all other land cover)  
**GLCM_Pan**   | Num    | Grey level mean texture (Pan band)  
**Mean_Green** | Num    | Mean green value  
**Mean_Red**   | Num    | Mean red value  
**Mean_NIR**   | Num    | Mean near infra-red value  
**SD_Pan**     | Num    | Standard deviation (Pan band)  

Table 1: Variable descriptions

**Known preceding interventions and pre-processing:**  

In its original form, the data is made up of two separate .csv files which consist of one training set of 4338 observations and one test set of 500 observations.

**Undertaken interventions and pre-processing:**  

* Remove observation 480 from imported test dataset: This observation appeared to be a significant outlier.   
* Merge test and train datasets into single dataset.  
* Split the merged dataset into a training set comprising 80% of observations and a test 
set comprising 20% of observations.  

*Note: Scaling of observations was not used.*  

Accordingly, the training dataset consists of 3,870 observations. 3,658 of which are classified
as disease free ('n') and 212 are classified as diseased ('w').  

The test dataset consists of 968 observations. 919 of which are classified as disease free ('n') 
and 49 are classified as diseased ('w').

```{r libraries, include=FALSE}
library(visNetwork)
library(sparkline)
library(rpart)
library(randomForest)
library(caret)
library(e1071)
library(tidyverse)
library(tinytex)
```


```{r set-data, include=FALSE}
setwd("/Users/temueracavanagh/Documents/Documents/Data Science/STAT6020 - Predictive Analytics/Project")

wilt = read.csv("wilt_training.csv", stringsAsFactors = TRUE)
wilt_test = read.csv("wilt_testing.csv", stringsAsFactors = TRUE)

# Remove row 480 from test 
wilt_test = wilt_test[-c(480),]

# Merge data sets
wilt = rbind(wilt, wilt_test)

# Check merged data
wilt[!complete.cases(wilt),] # No missing values
dim(wilt) # Check dimensions
nlevels(wilt$class) # Check class factor levels
table(wilt$class) # Check classifications

# Split train/test sets
set.seed(0) # Set seed for reproducibility 

# Select 80% of data as sample from total 'n' rows of the data  
sample = sample.int(n = nrow(wilt), size = floor(.8*nrow(wilt)), replace = F)
wilt_train = wilt[sample, ]
wilt_test = wilt[-sample, ]
```



***
### Methods:

R version

```{r version, echo=FALSE}
#RStudio.Version()$version
R.Version()$version.string
```

#### Decision Trees

```{r decision-tree}
class.rp = rpart(class ~ ., data = wilt_train)
```

```{r decision-treevis, echo=FALSE, fig.height = 3, fig.width = 3, fig.align = "center"}
visTree(class.rp, main = "Wilt Decision Tree",width = "80%",  height = "450px")
```
Figure 2: Wilt decision tree

```{r tree-train, include=FALSE}
rp.train = predict(class.rp, wilt_train, type = "class")
rp.train.matrix = confusionMatrix(rp.train, wilt_train$class)
rp.train.matrix$overall['Accuracy']
rp.train.matrix$byClass['Specificity']
```

```{r tree-test, include=FALSE}
# Predict on test set
rp.predict = predict(class.rp, wilt_test, type = "class")
rp.matrix = confusionMatrix(rp.predict, wilt_test$class)
rp.matrix$overall['Accuracy']
rp.matrix$byClass['Specificity']
```

Perform cross-validation and pruning of decision tree
```{r tree-cv, results='hide'}
set.seed(0)
tree.class.cv = printcp(class.rp) # Create matrix
min_misclass = which.min(tree.class.cv[, "xerror"]) # Find cost-complexity parameter with min CV error
# Resolve ties using the simplest model 
while(min_misclass < length(tree.class.cv[ , "xerror"]) &&
      tree.class.cv[min_misclass, "xerror"]==tree.class.cv[min_misclass+1, "xerror"]){
  min_misclass = min_misclass + 1}
CP = tree.class.cv[min_misclass, "CP"] # Cost-complexity corresponding to min CV error
tree.class.pruned = prune(tree = class.rp, cp = CP) # Prune best tree corresponding to min CV error
```

```{r tree-vis, echo=FALSE}
visTree(tree.class.pruned, main = "Pruned Decision Tree", width = "80%",  height = "450px")
```
Figure 3: Pruned decision tree

```{r prune-train, include=FALSE}
rp.prune.train.predict = predict(tree.class.pruned, wilt_train, type = "class")
prune.train.matrix = confusionMatrix(rp.prune.train.predict, wilt_train$class)
prune.train.matrix$overall['Accuracy']
prune.train.matrix$byClass['Specificity']
```

```{r prune-predict, include=FALSE}
rp.prune.predict = predict(tree.class.pruned, wilt_test, type = "class")
prune.matrix = confusionMatrix(rp.prune.predict, wilt_test$class)
prune.matrix$overall['Accuracy']
prune.matrix$byClass['Specificity']
```

#### Random Forests

```{r random-forest, results="hide"}
set.seed(0)
forest.train = randomForest(class ~ ., data = wilt_train, mtry = 5, importance = TRUE, ntree = 1000)
```

```{r, echo=FALSE}
forest.train
```

```{r plot-forest, echo=FALSE}
plot(forest.train, main = "Random Forest Error Rate")
```
Figure 4: Random forrest error rate

```{r, echo=FALSE}
varImpPlot(forest.train, main = "Importance of Variables in Random Forest")
```
Figure 5: Importance of Variables in Random Forest

```{r train-forest, include=FALSE}
forest.predict = predict(forest.train, wilt_train, type = "class")
forest.train.matrix = confusionMatrix(forest.predict, wilt_train$class)
forest.train.matrix$overall['Accuracy']
forest.train.matrix$byClass['Specificity']
```

```{r predict-forest, include=FALSE}
forest.test = predict(forest.train, wilt_test, type = "class")
forest.matrix = confusionMatrix(forest.test, wilt_test$class)
forest.matrix$overall['Accuracy']
forest.matrix$byClass['Specificity']
```

There is no requirement for cross-validation when using random forests (Breiman, 2001).

#### Support Vector Machines

```{r svm}
svm_class = svm(class~., data=wilt_train)
```

```{r, echo=FALSE}
svm_class
```

```{r}
plot(svm_class, data = wilt_train, Mean_Green~Mean_Red, 
     slice = list(GLCM_pan=1, Mean_NIR=1, SD_pan=1), xaxs="i")
```
Figure 6: Visualisation of SVM classifier

```{r svm-train, include=FALSE}
svm_training_pred = predict(svm_class, wilt_train)
svm.train.matrix = confusionMatrix(svm_training_pred, wilt_train$class)
svm.train.matrix$overall['Accuracy']
svm.train.matrix$byClass['Specificity']
```

```{r svm-test, include=FALSE}
svm_predict = predict(svm_class, wilt_test)
svm.test.matrix = confusionMatrix(svm_predict, wilt_test$class)
svm.test.matrix$overall['Accuracy']
svm.test.matrix$byClass['Specificity']
```

**Linear kernel Support Vector Machine**

```{r svm-linear}
set.seed(0)
tune.out=tune(svm, class~., data=wilt_train, kernel="linear",
              ranges=list(cost=c(0.001,0.01,0.1,1,10,100))) 
bestmod = tune.out$best.model
```
```{r, echo=FALSE}
bestmod
```

```{r svm-lineartrain, include=FALSE}
bestmod_train = predict(bestmod, wilt_train)
svm.linear.matrix = confusionMatrix(bestmod_train, wilt_train$class)
svm.linear.matrix$overall['Accuracy']
svm.linear.matrix$byClass['Specificity']
```

```{r svm-lineartest, include=FALSE}
bestmod_test = predict(bestmod, wilt_test)
svm.linear.test.matrix = confusionMatrix(bestmod_test, wilt_test$class)
svm.linear.test.matrix$overall['Accuracy']
svm.linear.test.matrix$byClass['Specificity']
```


**Radial kernel Support Vector Machine**

```{r svm-radial}
set.seed(0)
radtune.out=tune(svm, class~., data=wilt_train, kernel='radial',
                 ranges=list(cost=c(0.001,0.01,0.1,1,10,100),
                             gamma=c(0.01,0.1,1,2,3,4)))
bestmodrad = radtune.out$best.model
```

```{r, echo=FALSE}
bestmodrad
```

```{r svm-radialtrain, include=FALSE}
bestmodrad_train = predict(bestmodrad, wilt_train)
svm.radial.matrix = confusionMatrix(bestmodrad_train, wilt_train$class)
svm.radial.matrix$overall['Accuracy']
svm.radial.matrix$byClass['Specificity']
```

```{r svm-radialtest, include=FALSE}
bestmodrad_test = predict(bestmodrad, wilt_test)
svm.radial.test.matrix = confusionMatrix(bestmodrad_test, wilt_test$class)
svm.radial.test.matrix$overall['Accuracy']
svm.radial.test.matrix$byClass['Specificity']
```


***
### Results and Discussion:

Table 2 below shows the classification results from the previously highlighted methods.
Accuracy refers to the classification accuracy of a method when classifying observations across
both of the predictive classes.
Specificity refers to the classification accuracy of a method when classifying observations of
the class 'wilt' only.

```{r results, include=FALSE}
labs = data.frame(Method=c("Decision Tree",
                           "Pruned Decision Tree",
                           "Random Forest",
                           "Support Vector Machine (OOB)",
                           "Linear Support Vector Machine",
                           "Radial Support Vector Machine"))

train_accuracy = data.frame(Train.Accuracy=c(rp.train.matrix$overall['Accuracy'],
                                 prune.train.matrix$overall['Accuracy'],
                                 forest.train.matrix$overall['Accuracy'],
                                 svm.train.matrix$overall['Accuracy'],
                                 svm.linear.matrix$overall['Accuracy'],
                                 svm.radial.matrix$overall['Accuracy']))

train_specificity = data.frame(Train.Specificity=c(rp.train.matrix$byClass['Specificity'],
                                       prune.train.matrix$byClass['Specificity'],
                                       forest.train.matrix$byClass['Specificity'],
                                       svm.train.matrix$byClass['Specificity'],
                                       svm.linear.matrix$byClass['Specificity'],
                                       svm.radial.matrix$byClass['Specificity']))

accuracy = data.frame(Test.Accuracy=c(rp.matrix$overall['Accuracy'],
                                 prune.matrix$overall['Accuracy'],
                                 forest.matrix$overall['Accuracy'],
                                 svm.test.matrix$overall['Accuracy'],
                                 svm.linear.test.matrix$overall['Accuracy'],
                                 svm.radial.test.matrix$overall['Accuracy']))

specificity = data.frame(Test.Specificity=c(rp.matrix$byClass['Specificity'],
                                       prune.matrix$byClass['Specificity'],
                                       forest.matrix$byClass['Specificity'],
                                       svm.test.matrix$byClass['Specificity'],
                                       svm.linear.test.matrix$byClass['Specificity'],
                                       svm.radial.test.matrix$byClass['Specificity']))

results_table = cbind(labs, train_accuracy, train_specificity, accuracy, specificity)
```

```{r, echo=FALSE}
results_table
```
Table 2: Method results   

From the results obtained through the assessed methods, the radial support
vector machine method has recorded the equal highest accuracy (99.07%) and the highest 
specificity (95.92%) on the test dataset. The random forest method recorded the second highest 
accuracy (99.07%) and specificity (91.84%) on the test dataset, where 1000 trees where used (Figure 4).
Decision tree methods also performed comparatively well with accuracy and specificity results of 98.35% 
and 85.71% respectively for the un-pruned decision tree (Figure 2), and 98.14% and 81.63% respectively 
for the pruned decision tree (Figure 3). This is an observable reduction in specificity in the pruned 
decision tree in trade for a relatively minor simplification in the resulting decision tree model (16 
terminal nodes versus 13 terminal nodes when pruned).
Linear support vector machines performed with less accuracy and specificity when compared to 
radial kernel machines. However, given factors such as the importance of variables shown in Figure 5, 
and the resulting plot of these variables shown Figure 6, the classification potential for the linear 
support vector machines required analysis.
It is clear from analysis of the results that all methods recorded similar accuracy results
when each model was applied on the test dataset. As such, it can be stated that no model in
particular could be described to have been over-fitted to the training dataset.
It is also clear from analysis that all methods tended to perform with less precision
when specificity was considered. Assessing the precision in specificity of a method is 
important when consideration is given to the potential negative consequences of a false-negative
wilt disease classification. On this basis, specificity as a determining factor in the overall
assessment of the classification accuracy simply cannot be ignored.

***
### Conclusions:

In conclusion, this project has assessed and compared decision trees, random forest
ensembles for decision trees and support vector machines in the context of identifying 
diseased trees in the wilt dataset. A radial kernel support vector machine was identified
as a highly accurate and specific classifying method for the detection of wilt diseased
trees. In future, consideration should be given to the computational efficiency of the 
radial kernel support vector machine in comparison with other methods which performed accurately.

***
### References:

Breiman, L. (2001). "Random Forests", https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf

Food and Agriculture Organization of the United Nations Regional Office for Asia and the Pacific.
(2010). “Japan Forestry Outlook Study.” In Asia-Pacific Forestry Sector Outlook Study II Working
Paper Series, 9–10. Working Paper No. APFSOS II/WP/2010/30.

Johnson, B., Tateishi, R., Hoan, N. (2013). "A hybrid pansharpening approach and multiscale object-based image analysis for mapping diseased pine and oak trees", International Journal of Remote Sensing, 34 (20), 6969-6982. 

Garrett, K., Dendy, S., Frank, E., Rouse, M., Travers, M., (2006) “Climate change effects on plant disease: genomes to ecosystems,” Annual Review of Phytopathology, Vol. 44, pp. 489–509, 2006.

***
### Appendices:

Dataset source: http://archive.ics.uci.edu/ml/datasets/wilt
