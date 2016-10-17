library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(data.table)
library(Matrix)
library(xgboost)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

TRAIN = "../input/train.csv"
TEST = "../input/test.csv"
SUBMISSION = "../input/sample_submission.csv"

train_data = fread(TRAIN)
test_data = fread(TEST)

X_label = log(train_data[,'loss', with = FALSE])[['loss']]
train_data[, c('id', 'loss') := NULL]
test_data[, c('id') := NULL]

ntrain = nrow(train_data)
train_test_data = rbind(train_data, test_data)

features = names(train_data)

for (x in features) {
  if (class(train_test_data[[x]])=="character") {
    levels <- unique(train_test_data[[x]])
    train_test_data[[x]] <- as.integer(factor(train_test_data[[x]], levels=levels))
  }
}

X_train = train_test_data[1:ntrain,]
X_test = train_test_data[(ntrain+1):nrow(train_test_data),]

#model_xgb <- xgboost(data=as.matrix(X_train), label=as.matrix(X_label), booster="gblinear", objective="reg:linear", nrounds=1000, eta=0.05, max_depth=6, subsample=0.75, colsample_bytree=0.8, min_child_weight=1)
model_xgb <- xgboost(data=as.matrix(X_train), label=as.matrix(X_label), booster="gblinear", objective="reg:linear", nrounds=5000, eta=0.05, max_depth=8, subsample=0.9, colsample_bytree=0.9, min_child_weight=2)
submission <- read.csv("../input/sample_submission.csv")
submission$loss = exp(predict(model_xgb, as.matrix(X_test)))
write.csv(submission, "XGB_starter.csv", row.names = FALSE)

# Any results you write to the current directory are saved as output.