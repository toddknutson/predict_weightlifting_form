# Practical Machine Learning Project
Todd Knutson  
July 22, 2014  
# Introduction
"Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz38Egi8PXw"



Six different weight lifters (user_name) were monitored completing the exercise in five different weight lifting styles (classe) labeled: A-E. Class A style is the correct way to complete the exercise, and classes B-E are common mistakes.



# Results
Here, I will start the analsis.


## Data Processing

### Download Data


```r
# Create directory for raw data download
if(!file.exists("Raw Data")){
  dir.create("Raw Data")
}

# Download raw data
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
## [1] "Tue Jul 22 16:51:02 2014"
```

### Import Data

Load the data into R.

```r
train <- read.csv("Raw Data/train.csv", header = TRUE)
test <- read.csv("Raw Data/test.csv", header = TRUE)
```



Look at the dimensions of both training and test data sets. The test data should have the same number of variables (160), but many fewer observations.

```r
dim(train); dim(test)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```



Examine the training data variables. Indeed, six participants were measured completing the exercises in five different styles (A-E), with an approximatley equal number of observations per person and lifting style.

```r
table(train$user_name); table(train$classe)
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

### Subset the training data set for cross validation


Split the training set data into two groups, one group for creating the prediction algorithm and the other for cross validation.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(12345)
inTrain1 <- createDataPartition(train$classe, p = 3/4)[[1]]
train1 <- train[inTrain1, ]
train2 <- train[-inTrain1, ]
```




## Build Prediction Model

# Conclusions

# References
Citation for original experiment and source files: http://groupware.les.inf.puc-rio.br/har



