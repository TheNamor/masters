
library(ggplot2)
library(e1071)
library(randomForest)
library(class)

features = data.frame(read.csv("~/Data/features.csv"))

features = subset(features, select=-c(X..iso, X..end, radius, radius.90, avg.shortest.path, num.twos))

features = as.data.frame(scale(features))

features = cbind(features, label = as.factor(c(
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0
)))

features = cbind(features, label = as.factor(c(
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0
)))

max(features$avg.deg)

hist(features$avg.deg, breaks=28)

qplot(features$avg.deg, features$avg.clus, color=labels)

set.seed(101913)

knn_accs = data.frame(matrix(ncol = 10, nrow = 0))
colnames(knn_accs) = c(1:10)
knn_table_agg = as.table(rbind(c(0,0,0), c(0,0,0), c(0,0,0)))

svm_accs = data.frame(matrix(ncol = 3, nrow = 0))
colnames(svm_accs) = c("radial", "linear", "poly")
svm_table1 = as.table(rbind(c(0,0,0), c(0,0,0), c(0,0,0)))
svm_table2 = as.table(rbind(c(0,0,0), c(0,0,0), c(0,0,0)))
svm_table3 = as.table(rbind(c(0,0,0), c(0,0,0), c(0,0,0)))


rf_accs = data.frame(matrix(ncol = 1, nrow = 0))
colnames(rf_accs) = c(1:1)
rf_table = as.table(rbind(c(0,0,0), c(0,0,0), c(0,0,0)))

for (j in 1:20) {

  train = sample(1:nrow(features), nrow(features)*0.8)
  features_train = features[train,]
  features_test = features[-train,]
  
  svm1 = svm(label~., data=features_train)

  pred1 = predict(svm1, features_test)
  
  table1 = table(Predicted=pred1, Actual=features_test$label)
  table1
  svm_table1 = svm_table1 + table1
  
  model1_accuracy = sum(diag(table1))/sum(table1)
  model1_accuracy
  
  missrate = 1 - model1_accuracy
  missrate
  
  svm2 = svm(label~., data=features_train, kernel="linear")

  pred2 = predict(svm2, features_test)
  
  table2 = table(Predicted=pred2, Actual=features_test$label)
  table2
  svm_table2 = svm_table2 + table2
  
  model2_accuracy = sum(diag(table2))/sum(table2)
  model2_accuracy
  
  missrate2 = 1 - model2_accuracy
  missrate2
  
  svm3 = svm(label~., data=features_train, kernel="polynomial", degree=3)

  pred3 = predict(svm3, features_test)
  
  table3 = table(Predicted=pred3, Actual=features_test$label)
  table3
  svm_table3 = svm_table3 + table3
  
  model3_accuracy = sum(diag(table3))/sum(table3)
  model3_accuracy
  
  missrate3 = 1 - model3_accuracy
  missrate3
  
  svm_accs = rbind(svm_accs, c(model1_accuracy, model2_accuracy, model3_accuracy))
  
  #accs = c()
  #for (i in 1:5) {
  #  ntree = c(200, 300, 500, 600, 750)[i]
  #  rf = randomForest(label ~ ., features, ntree = ntree)
  #  accs[i] = 1 - (rf$confusion[1,2] + rf$confusion[2,1])/138
  #}
  
  #accs
  
  rf1 = randomForest(label ~ ., features_train, ntree=200, mtry=3)
  #rf1
  
  #1 - (rf1$confusion[1,2] + rf1$confusion[2,1])/138
  
  pred4 = predict(rf1, features_test)
  
  table4 = table(Predicted=pred3, Actual=features_test$label)
  table4
  
  rf_table = rf_table + table4
  
  model4_accuracy = sum(diag(table4))/sum(table4)
  model4_accuracy
  
  rf_accs = rbind(rf_accs, model4_accuracy)
  
  missrate4 = 1 - model4_accuracy
  missrate4
  
  #varImpPlot(rf1)
  
  accs = c()
  for (i in 1:10) {
    knn1 = knn(features_train, features_test, features_train$label, k=i)
    knn_table = table(knn1, features_test$label)
    accs[i] = sum(diag(knn_table))/sum(knn_table)
  }
  
  knn_accs = rbind(knn_accs, accs)
  
  knn1 = knn(features_train, features_test, features_train$label, k=1)
  knn_table = table(knn1, features_test$label)
  knn_table_agg = knn_table_agg + knn_table
  sum(diag(knn_table))/sum(knn_table)
}

for (i in 1:10) {
  print(mean(knn_accs[[i]]))
}

knn_table_agg

for (i in 1:3) {
  print(mean(svm_accs[[i]]))
}

mean(rf_accs[[1]])

rf_table

svm_table1
