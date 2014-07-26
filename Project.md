# Using Machine Learning to Predict Correct Weightlifting Form
July 22, 2014  
# Introduction

In recent years, many new personal activity devices have been used by persons interested in tracking their movement over the course of the day. In this study, we will utilize these types of devices (accelerometers) to investigate movement data from a person performing a dumbbell weightlifting exercise. Six volunteers wore four accelerometers strapped to the dumbbell, forearm, upper-arm, and belt while performing the exercise. The volunteers completed the Unilateral Dumbbell Biceps Curl in five different fashions: (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).   

We will use this data to train a machine learning algorithm that can predict which version of the exercise (A-E) that 20 unknown test subjects completed.   

First, we will import the training and test datasets provided, subset the training data into sub-training and sub-testing sets for cross validation, and finally train a boosting algorithm using tree classification on the sub-training data. After adjusting the algorithm and estimating the out-of-sample error based on cross validation, the classifier will be used on the test data. The predictions will be submitted for evaluation.




Six different weight lifters (user_name) were monitored completing the exercise in five different weight lifting styles (classe) labeled: A-E. Class A style is the correct way to complete the exercise, and classes B-E are common mistakes.



# Results



## Data Processing

### Download Data


```r
# Create directory for raw data download
if(!file.exists("Raw Data")){
  dir.create("Raw Data")
}
# Download raw data from Coursera website
trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(trainUrl, destfile = "./Raw Data/train.csv", method = "curl")
download.file(testUrl, destfile = "./Raw Data/test.csv", method = "curl")
list.files("./Raw Data")
```

```
## [1] "test.csv"  "train.csv"
```

```r
dateDownloaded <- date()
dateDownloaded
```

```
## [1] "Fri Jul 25 15:43:43 2014"
```

### Import Data

Load the two provided datasets into R.

```r
train <- read.csv("Raw Data/train.csv", header = TRUE, na.strings = "", colClasses = "character", row.names = 1)
test <- read.csv("Raw Data/test.csv", header = TRUE, na.strings = "", colClasses = "character", row.names = 1)
```





Look at the dimensions of both training and test datasets. The test data should have the same number of variables (159), but many fewer observations.

```r
dim(train); dim(test)
```

```
## [1] 19622   159
```

```
## [1]  20 159
```



### Make Training Data Tidy
Clean the raw training data to make it tidy. Identify variables with missing values and remove them. 

```r
# Set options to a negative number so warning messages are not printed (warnings will occur due to numeric coercion below)
options(warn = -1)
# Keep first 6 columns as "character" class, and the rest "numeric"" class
train_tidy <- cbind(train[, 1:5], sapply(train[, 6:158], as.numeric))

# Add the "classe" variable back to the dataset as a factor
train_tidy$classe <- factor(train$classe)
# Convert other variables to factors
train_tidy$user_name <- factor(train_tidy$user_name)
train_tidy$raw_timestamp_part_1 <- factor(train_tidy$raw_timestamp_part_1)
train_tidy$raw_timestamp_part_2 <- factor(train_tidy$raw_timestamp_part_2)
train_tidy$new_window <- factor(train_tidy$new_window)
train_tidy$cvtd_timestamp <- NULL

# For each variable, determine the number of NA values in each column
na <- vector()
for (i in 1:length(colnames(train_tidy))) {
na[i] <- length(which(is.na(train_tidy[, i])))
}
na
```

```
##   [1]     0     0     0     0     0     0     0     0     0 19226 19248
##  [12] 19622 19225 19248 19622 19216 19216 19226 19216 19216 19226 19216
##  [23] 19216 19226 19216 19216 19216 19216 19216 19216 19216 19216 19216
##  [34] 19216     0     0     0     0     0     0     0     0     0     0
##  [45]     0     0     0 19216 19216 19216 19216 19216 19216 19216 19216
##  [56] 19216 19216     0     0     0     0     0     0     0     0     0
##  [67] 19294 19296 19227 19293 19296 19227 19216 19216 19216 19216 19216
##  [78] 19216 19216 19216 19216     0     0     0 19221 19218 19622 19220
##  [89] 19217 19622 19216 19216 19221 19216 19216 19221 19216 19216 19221
## [100]     0 19216 19216 19216 19216 19216 19216 19216 19216 19216 19216
## [111]     0     0     0     0     0     0     0     0     0     0     0
## [122]     0 19300 19301 19622 19299 19301 19622 19216 19216 19300 19216
## [133] 19216 19300 19216 19216 19300     0 19216 19216 19216 19216 19216
## [144] 19216 19216 19216 19216 19216     0     0     0     0     0     0
## [155]     0     0     0     0
```

```r
# Keep only variables (columns) from training data without any NA values
train_tidy <- train_tidy[, na == 0]
```


Examine the final training data variables. Indeed, six participants were measured completing the exercises in five different styles (A-E), with an approximately equal number of observations per person and lifting style.

```r
table(train_tidy$user_name); table(train_tidy$classe)
```

```
## 
##   adelmo carlitos  charles   eurico   jeremy    pedro 
##     3892     3112     3536     3070     3402     2610
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```



### Explore the variables

Explore the variables in the data. Make a few plots comparing the numeric accelerator variables to the weightlifting class variable. Look for possible correlations between these variables that may suggest good predictor variables of the weightlifting class. There appears to be some unique clusters of points between the measurement variables and the class variable, especially in the "arm" and "forearm" variables. These will likely contribute more information in the classifier training.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
p1 <- featurePlot(x = train_tidy[, c("roll_belt", "pitch_belt", "yaw_belt")], y = train_tidy$classe, plot = "pairs") 
p2 <- featurePlot(x = train_tidy[, c("roll_arm", "pitch_arm", "yaw_arm")], y = train_tidy$classe, plot = "pairs")
p3 <- featurePlot(x = train_tidy[, c("roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell")], y = train_tidy$classe, plot = "pairs")
p4 <- featurePlot(x = train_tidy[, c("roll_forearm", "pitch_forearm", "yaw_forearm")], y = train_tidy$classe, plot = "pairs")
library(gridExtra)
```

```
## Loading required package: grid
```

```r
grid.arrange(p1, p2, p3, p4, ncol = 2, main = "Comparison of Accelerometer Variables\n(colored by weightlifting class variable)")
```

![plot of chunk unnamed-chunk-6](./Project_files/figure-html/unnamed-chunk-6.png) 


## Build a Prediction Model


The training data will be split into two groups, one group for training the prediction model and the other for cross validation to estimate the out-of-sample error. A *random sampling* method will be used to split the training data into these two groups, where the model will be built using the sub-training set, and tested on the sub-testing set. The process will be repeated multiple times and the prediction model error will be averaged among each cross validation iteration. For each subsetting iteration of training data, both sets will contain equal proportions of each exercise style (A-E).    

In our preliminary plotting, looking for correlated variables, we did not find any strong single predictor variables, we will consolidate all variables using Principal Components Analysis (PCA). Within each iteration of the cross validation, all the numeric variables will first be pre-processed using PCA, with a threshold setting to capture 90% of the variability in the variables. Then, using the resulting PCA variables, a boosting model will be fitted using trees via the "gbm" package. The results from our cross validation will be collected an a confusion matrix can be analyzed.   

```r
library(caret)
library(gbm)

# Create empty lists to collect results from loop
inTrain <- list()
sub_train <- list()
sub_test <- list()
preProc <- list()
trainPC <- list()
modelFit_train <- list()
testPC <- list()
results <- list()
# Loop the number of times cross validation should occur
#i = 1
for (i in 1:2) {
    # For cross validation, subsplit training data into new groups: sub_train and sub_test 
    set.seed(25 * i)
    inTrain[[i]] <- createDataPartition(train_tidy$classe, p = 3/4, list = FALSE)
    sub_train[[i]] <- train_tidy[inTrain[[i]], ]
    sub_test[[i]] <- train_tidy[-inTrain[[i]], ]

    metadata_cols <- c(1:5)
    sub_train[[i]] <- sub_train[[i]][, -metadata_cols]
    sub_test[[i]] <- sub_test[[i]][, -metadata_cols]
    

    # Preprocess data using principal componenets
    # Column 53 contains the variable we wish to predict (classe) and will be removed
    preProc[[i]] <- preProcess(sub_train[[i]][, -53], method = "pca", thresh = 0.9)
    trainPC[[i]] <- predict(preProc[[i]], sub_train[[i]][, -53])
    
    
    
    
    # Prediction using boosting with trees
    # Fit the model
    modelFit_train[[i]] <- train(sub_train[[i]][, "classe"] ~ ., method = "gbm", data = trainPC[[i]], verbose = FALSE)
    # Calculate get the principal componenets for the sub_test set
    testPC[[i]] <- predict(preProc[[i]], sub_test[[i]][, -53])
    # Calculate predictions on the sub_test set
    results[[i]] <- confusionMatrix(sub_test[[i]][, "classe"], predict(modelFit_train[[i]], testPC[[i]]))

}
```


### Estimate Out-of-Sample Error

Compare the cross validated results from the sub-test groups. Find the *mean* Accuracy, Sensitivity, and Specificity from each cross validation iteration on the sub-test data.

```r
sub_test_accuracy <- mean(c(results[[1]]$overall[1], results[[2]]$overall[1]))
sub_test_sensitivity <- apply(data.frame(results[[1]]$byClass[, 1], results[[2]]$byClass[, 1]), 1, mean)
sub_test_specificity <- apply(data.frame(results[[1]]$byClass[, 2], results[[2]]$byClass[, 2]), 1, mean)
# Print out these values
sub_test_accuracy
```

```
## [1] 0.8056
```

```r
sub_test_sensitivity
```

```
## Class: A Class: B Class: C Class: D Class: E 
##   0.8394   0.7728   0.7083   0.7936   0.9178
```

```r
sub_test_specificity
```

```
## Class: A Class: B Class: C Class: D Class: E 
##   0.9494   0.9382   0.9599   0.9570   0.9521
```



After running multiple cross validation prediction examples, the accuracy is approximately 80%, Sensitivity between 70-90% for each weightlifting class, and Specificity between 93-95% for each weightlifting class. With these values, I am comfortable to run my model on the actual test data.

## Run Prediction Model on Test Data


### Make Test Data Tidy


```r
# Keep first 6 columns as "character"" class, and the rest "numeric"" class
test_tidy <- cbind(test[, 1:5], sapply(test[, 6:158], as.numeric))

# Add back the "problem_id" variable
test_tidy$problem_id <- as.integer(test$problem_id)
# Convert variables to factors
test_tidy$user_name <- factor(test_tidy$user_name)
test_tidy$raw_timestamp_part_1 <- factor(test_tidy$raw_timestamp_part_1)
test_tidy$raw_timestamp_part_2 <- factor(test_tidy$raw_timestamp_part_2)
test_tidy$new_window <- factor(test_tidy$new_window)
test_tidy$cvtd_timestamp <- NULL


# Keep only the same variables (columns) in the test data that were kept from the training data
test_tidy <- test_tidy[, na == 0]
```

### Train model with all training data


```r
# For cross validation, subsplit training data into new groups: sub_train and sub_test 
set.seed(12345)

metadata_cols <- c(1:5)
final_train <- train_tidy[, -metadata_cols]
final_test <- test_tidy[, -metadata_cols]

# Preprocess data using principal componenets
# Column 53 contains the variable we wish to predict (problem_id) and will be removed
final_preProc <- preProcess(final_train[, -53], method = "pca", thresh = 0.9)
final_trainPC <- predict(final_preProc, final_train[, -53])

# Prediction using boosting with trees
# Fit the model
final_modelFit_train <- train(final_train[, "classe"] ~ ., method = "gbm", data = final_trainPC, verbose = FALSE)
```


### Make Predictions on Test Data wtih Model



```r
# Calculate get the principal componenets for the test set
final_testPC <- predict(final_preProc, final_test[, -53])
# Make predictions on the sub_test set
final_results <- predict(final_modelFit_train, final_testPC)
```


```r
library(knitr)
kable(data.frame(Unknown = seq(1:20), Prediction = final_results), row.names = NA, align = "c")
```

```
## 
## 
## | Unknown | Prediction |
## |:-------:|:----------:|
## |    1    |     C      |
## |    2    |     A      |
## |    3    |     A      |
## |    4    |     A      |
## |    5    |     A      |
## |    6    |     E      |
## |    7    |     D      |
## |    8    |     B      |
## |    9    |     A      |
## |   10    |     A      |
## |   11    |     A      |
## |   12    |     C      |
## |   13    |     B      |
## |   14    |     A      |
## |   15    |     E      |
## |   16    |     A      |
## |   17    |     A      |
## |   18    |     B      |
## |   19    |     D      |
## |   20    |     B      |
```









# Conclusions

In this study, we have utilized a training dataset of persons weightlifting in five different methods (classes A-E). From the training data, we preprocessed the variables using PCA and then employed a machine learning algorithm using boosting with decision trees to create a model. The out-of-sample error was estimated using cross validation within the training data via random sampling method, with about 80% accuracy. Finally, the algorithm was applied to the test dataset of 20 unknowns for prediction. After uploading the results, 15/20 unknowns were correctly identified (75%). This makes sense because the out-of-sample error will almost always be lower than in-sample error (80%). Therefore, with this algorithm, the "fashion" of weightlifting can be predicted with 75% accuracy using the accelerometer variables.

# References
Citation for the original experiment and source files: http://groupware.les.inf.puc-rio.br/har



