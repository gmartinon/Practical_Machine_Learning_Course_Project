# Prediction Model on the Quality of Execution of Bodybuilding Movements

## Executive Summary

We use data from [this research paper][1] to predict "how well" a bodybuilding movement is executed. Each training execution is attributed a note (A, B, C, D, E) and our goal is to predict, with the help of data harvested by accelerometers and gyroscopes, the notes corresponding to testing executions. For this purpose, we find that a random forest machine learning algorithm yields an overall accuracy of 99%.

## Loading the Data
We first load the packages and data.

```r
if (!require("pacman")) install.packages("pacman")
p_load(ggplot2, caret, parallel, doParallel, foreach, knitr)
```

```r
trainurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainfile <- "training.csv"
testfile <- "testing.csv"
if (!file.exists(trainfile)) download.file(trainurl, trainfile)
if (!file.exists(testfile)) download.file(testurl, testfile)

data <- read.csv(trainfile)
```
Among the columns of the data set, several columns feature more than 97% NA, and other are character factors. We thus decide to put them aside and focus on numerical accelerometer data.

```r
mycols <- grep("^(total_accel|accel|magnet|gyros|yaw|roll|pitch)_(belt|arm|forearm|dumbbell)",
                 names(data), value = TRUE)
data <- data[c("classe",mycols)]
```
We process in the same way the validation set, that is to be used for the project quizz.

```r
valid <- read.csv(testfile)
valid <- valid[mycols]
```

## Partitioning the data
We then partition the data according to a training and testing set, with around 60% of data used for training and 40% for testing.

```r
inTrain <- createDataPartition(data$classe, p = 0.6, list = FALSE)
training <- data[inTrain,]
testing <- data[-inTrain,]
```

## Random Forest Prediction Model
We then train the machine on the training data set with a random forest algorithm. Indeed, it appears to be one of most accurate algorithms to date, even if it is more computationally demanding.

In order to increase computation speed, we parallelise the computation accross four processors. We also pre-process the data via principal component analysis and retain only the first five components. We use an **out-of-bag (OOB) resampling method for cross-validation** and decrease the number of trees to 100 (default 500) to increase again computational speed. We have checked that these simplifications were not impacting (at all) the accuracy. The model is generated in about a minute on 2 GHz intel cores (i3).

```r
cluster <- makeCluster(4)
registerDoParallel(cluster)
fitControl <- trainControl(preProcOptions = list(pcaComp = 5),
                          method = "oob",
                          number = 5)
mdl <- train(classe ~ ., data = training, method = "rf", ntree = 100,
             preprocess = "pca",
             trControl = fitControl)
stopCluster(cluster)
registerDoSEQ()
```

## Results

We now apply our model to the testing data set. The confusion matrix and accuracy are given by

```r
cfmat <- confusionMatrix(testing$classe, predict(mdl, testing))
kable(cfmat$table,
      caption = "Confusion matrix on the testing data set with our trained model.")
```



Table: Confusion matrix on the testing data set with our trained model.

         A      B      C      D      E
---  -----  -----  -----  -----  -----
A     2231      1      0      0      0
B        6   1508      3      1      0
C        0     14   1345      9      0
D        0      0     17   1269      0
E        1      3      1      8   1429

```r
cfmat$overall[1]
```

```
## Accuracy 
## 0.991843
```
We thus have 99% accuracy, with sensitivity and specificity of the same order of magnitude. The out of sample OOB **error rate** is thus

```r
1 - cfmat$overall[[1]]
```

```
## [1] 0.008157023
```

Our prediction for the validation data set are then

```r
predict(mdl, valid)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## References

[The research team website on which this project is based][1]

[1]: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har
