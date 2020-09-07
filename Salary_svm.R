library(readr)
library(caret)
library(kernlab)
library(e1071)
library(ggplot2)

#Importing Dataset
Salary_train <- read.csv("C:\\Users\\91755\\Desktop\\Assignment\\15 - SVM\\SalaryData_Train.csv")
Salary_test <- read.csv("C:\\Users\\91755\\Desktop\\Assignment\\15 - SVM\\SalaryData_Test.csv")
attach(Salary_train)
attach(Salary_test)
head(Salary_train)
head(Salary_test)

#EDA and Statistical Analysis
sum(is.na(Salary_train))
sum(is.na(Salary_test))

str(Salary_train)
summary(Salary_train)
table(Salary_train$Salary)
class(Salary_train)
table(Salary_train$Salary)
table(Salary_test$Salary)

Salary_train$educationno <- factor(Salary_train$educationno)
Salary_test$educationno <- factor(Salary_test$educationno)

#Graphical Representation
plot(Salary_train$education, Salary_train$Salary)
plot(Salary_train$educationno, Salary_train$Salary)
plot(Salary_train$occupation, Salary_train$Salary)
plot(Salary_train$maritalstatus, Salary_train$Salary)

ggplot(data = Salary_train, aes(x=Salary, y=age)) + geom_boxplot()+ggtitle("Boxplot")
ggplot(data = Salary_train, aes(x=native, y=Salary, fill=Salary)) + geom_density(alpha=0.9, color="red")

#Model Building
kernals <- c("rbfdot", "polydot", "vanilladot", "besseldot", "anovadot")
acc_bag <- list()
pred <- list()
table <- list()
for (i in kernals) {
  model_bag <- ksvm(Salary~., data=Salary_train, kernel=i)
  pred_bag <- predict(model_bag, Salary_test)
  pred[[i]] <- (pred_bag) 
  table[[i]] <- confusionMatrix(pred_bag, Salary_test$Salary)
}
pred
table$rbfdot               #Accuracy = 85.21
table$polydot              #Accuracy = 84.61
table$vanilladot           #Accuracy = 84.64
table$besseldot            #Accuracy = 78.97
table$anovadot             #Accuracy = 78.15

#Final model
Model_rbf <- ksvm(Salary~., data=Salary_train, kernel="rbfdot")
predict_rbf <- predict(Model_rbf, Salary_test)
confusionMatrix(predict_rbf, Salary_test$Salary) #Accuracy = 85.21
plot(predict_rbf)
