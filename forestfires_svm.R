library(readr)
library(caret)
library(kernlab)
library(e1071)
library(ggplot2)
library(DataExplorer)
library(onewaytests)

#Importing Dataset
Forestfire_Data <- read.csv("C:\\Users\\91755\\Desktop\\Assignment\\15 - SVM\\forestfires.csv")
attach(Forestfire_Data)
View(Forestfire_Data)

#EDA and Statistical Analysis
sum(is.na(Forestfire_Data))
str(Forestfire_Data)
Forestfire <- Forestfire_Data[c("month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area", "size_category")]
attach(Forestfire)
head(Forestfire)
summary(Forestfire)
table(Forestfire$size_category)
str(Forestfire)

#Converting into Intergers
table(Forestfire$month)
Forestfire$month <- as.integer(factor(Forestfire$month, levels = c("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
                                      labels = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)))
table(Forestfire$day)
Forestfire$day <- as.integer(factor(Forestfire$day, levels = c("sun", "mon", "tue", "wed", "thu", "fri", "sat"),
                                    labels = c(1, 2, 3, 4, 5, 6, 7)))

#Normalisation of Data
normal <- function(x)
{
  return((x-min(x))/(max(x)-min(x)))
}
Forest <- as.data.frame(lapply(Forestfire[, -12], normal))
Forestfire_norm <- data.frame(Forest, size_category)
head(Forestfire_norm)
summary(Forestfire_norm)

#Graphical Representation
plot(Forestfire_norm)
plot_bar(Forestfire_norm)
plot_histogram(Forestfire_norm)
plot_correlation(Forestfire_norm)
ggplot(mapping= aes(x=Forestfire_norm$area), data=Forestfire_norm, color=Forestfire_norm$size_category)+geom_histogram()
qplot(temp,RH,data = Forestfire_norm,color=size_category,geom=c("boxplot"))
qplot(temp,RH,data = Forestfire_norm,color=size_category)
plot_density(Forestfire_norm)

#Data Splitting into Train and Test
set.seed(123)
split <- createDataPartition(Forestfire_norm$size_category, p=0.75, list = F)
Forest_train <- Forestfire_norm[split,]
Forest_test <- Forestfire_norm[-split,]
head(Forest_train)

#Model Building
#Model Building on Radial Basis kernel Using rbfdot
model_rdf <- ksvm(size_category~., data=Forest_train, kernel="rbfdot")
summary(model_rdf)
pred_rbf <- predict(model_rdf, Forest_test)
confusionMatrix(pred_rbf, Forest_test$size_category)                #Accuracy = 81.25%
plot(pred_rbf)

#Model Building on Polynomial kernel Using polydot
model_polydot <- ksvm(size_category~., data=Forest_train, kernel="polydot")
summary(model_polydot)
pred_ploydot <- predict(model_polydot, Forest_test)
confusionMatrix(pred_ploydot, Forest_test$size_category)            #Accuracy = 89.06%
plot(pred_ploydot)

#Model Building on Hyperbolic tangent kernel Using tanhdot
model_tanhdot <- ksvm(size_category~., data=Forest_train, kernel="tanhdot")
summary(model_tanhdot)
pred_tanhdot <- predict(model_tanhdot, Forest_test)
confusionMatrix(pred_tanhdot, Forest_test$size_category)         #Accuracy = 55.47

#Model Building on Bessel Kernel using besseldot
model_besseldot <- ksvm(size_category~., data=Forest_train, kernel="besseldot")
summary(model_besseldot)
pred_besseldot <- predict(model_besseldot, Forest_test)
confusionMatrix(pred_besseldot, Forest_test$size_category)      #Accuracy = 79.69

#Model Building on Laplacian Kernel using laplacedot
model_laplacedot <- ksvm(size_category~., data=Forest_train, kernel="laplacedot")
summary(model_laplacedot)
pred_laplacedot <- predict(model_laplacedot, Forest_test)
confusionMatrix(pred_laplacedot, Forest_test$size_category)    #Accuracy = 80.47

#Model Building on ANOVA RBF Kernel using anovadot
model_anovadot <- ksvm(size_category~., data=Forest_train, kerenel="anovadot")
summary(model_anovadot)
pred_anovadot <- predict(model_anovadot, Forest_test)
confusionMatrix(pred_anovadot, Forest_test$size_category)     #Accuracy = 81.25

#Model Building on Spline Kernel using splinedot
model_splinedot <- ksvm(size_category~., data=Forest_train, kernel="splinedot")
summary(model_splinedot)
pred_splinedot <- predict(model_splinedot, Forest_test)
confusionMatrix(pred_splinedot, Forest_test$size_category)    #Accuracy = 62.5

#Final Model
#Model Building on Linear kernel Using vanilladot
model_vanilladot <- ksvm(size_category~., data=Forest_train, kernel="vanilladot")
summary(model_vanilladot)
pred_vanilladot <- predict(model_vanilladot, Forest_test)
confusionMatrix(pred_vanilladot, Forest_test$size_category)       #Accuracy = 89.06%