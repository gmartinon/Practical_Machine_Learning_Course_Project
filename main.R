setwd("~/Data_Science/coursera/Practical_Machine_Learning/CourseProject")
if (!require("pacman")) install.packages("pacman")
p_load(ggplot2, caret, parallel, doParallel, foreach)

trainurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainfile <- "training.csv"
testfile <- "testing.csv"
if (!file.exists(trainfile)) download.file(trainurl, trainfile)
if (!file.exists(testfile)) download.file(testurl, testfile)

data <- read.csv(trainfile)
mycols <- grep("^(total_accel|accel|magnet|gyros|yaw|roll|pitch)_(belt|arm|forearm|dumbbell)",
                 names(data), value = TRUE)
data <- data[c("classe",mycols)]
valid <- read.csv(testfile)
valid <- valid[mycols]

set.seed(46354)
inTrain <- createDataPartition(data$classe, p = 0.6, list = FALSE)
training <- data[inTrain,]
testing <- data[-inTrain,]

cluster <- makeCluster(4)
registerDoParallel(cluster)
fitControl <- trainControl(preProcOptions = list(pcaComp = 5),
                          method = "oob",
                          number = 5,
                          allowParallel = TRUE)
system.time({
mdl <- train(classe ~ ., data = training, method = "rf", ntree = 100,
             preprocess = "pca",
             trControl = fitControl)
})
stopCluster(cluster)
registerDoSEQ()

confusionMatrix(testing$classe, predict(mdl, testing))$overall[1]
predict(mdl, valid)
