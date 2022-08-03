## The Dataset
movies <- read.csv("data/movies_metadata.csv")

str(movies)

### Processing the data

#### Loading the required packages:
if (!require("caret")) install.packages("caret"); library("caret")
if (!require("e1071")) install.packages("e1071"); library("e1071")
if (!require("kernlab")) install.packages("kernlab"); library("kernlab")
if (!require("rpart")) install.packages("rpart"); library("rpart")
if (!require("rpart.plot")) install.packages("rpart.plot"); library("rpart.plot")
if (!require("randomForest")) install.packages("randomForest"); library("randomForest")
if (!require("kableExtra")) install.packages("kableExtra"); library("kableExtra")
if (!require("readr")) install.packages("readr"); library("readr")
if (!require("knitr")) install.packages("knitr"); library("knitr")


bad_data_index <- which(movies$adult != 'TRUE' & movies$adult != 'FALSE')
movies <- movies[-c(bad_data_index),]

movies <- movies[,-c(1,4,6,7,8,9,10,11,13,14,15,16,17,18,21,22,23,24,25,26)]

movies <- na.exclude(movies)

movies$belongs_to_collection[movies$belongs_to_collection == ''] <- 0
movies$belongs_to_collection[movies$belongs_to_collection != 0] <- 1
movies <- transform(movies,belongs_to_collection = factor(belongs_to_collection, 
                                                          levels = c(0, 1), labels = c(0, 1)))

movies$budget[movies$budget==""] <- NA
movies$budget <- as.numeric(movies$budget)

movies$action <- 0
movies$action[movies$genres_adj=="Action"] <- 1
movies$action <- factor(movies$action)

movies$comedy <- 0
movies$comedy[movies$genres_adj=="Comedy"] <- 1
movies$comedy <- factor(movies$comedy)

movies$documentary <- 0
movies$documentary[movies$genres_adj=="Documentary"] <- 1
movies$documentary <- factor(movies$documentary)

movies$drama <- 0
movies$drama[movies$genres_adj=="Drama"] <- 1
movies$drama <- factor(movies$drama)

movies <- movies[,-c(3)]

movies$popularity <- as.numeric(movies$popularity)

movies$vote_average[movies$vote_average >= 7] <- "pos"
movies$vote_average[movies$vote_average < 7] <- 0
movies$vote_average[movies$vote_average == "pos"] <- 1
movies <- transform(movies,vote_average = factor(vote_average, levels = c(0, 1),
                                                 labels = c(0, 1)))

movies <- na.exclude(movies)

### Final dataset 

str(movies)

### Descriptive statistics

summary(movies)

varCount <- ncol(movies)
varNames <- names(movies)
factor_index <- which(lapply(movies, class) == "factor")

op <- par(mfrow = c(2, 6), cex = .5, mar = c(2.5, 2.5, 2.5, 2.5), mgp = c(1.4, .5, 0))
for (i in seq_along(varNames)) {
  if(i %in% factor_index) {
    barplot(table(movies[[i]]), main=varNames[i], ylab="Frequency")
  } else {
    hist(movies[[i]], freq=TRUE, main=varNames[i], xlab = "",)
  }
}
par(op)

op <- par(mfrow = c(2, 6), cex = .4, mar = c(2,3,2,2), mgp = c(1.5, .5, 0))
for(i in 1 : varCount) {
  if(i==6) {
    
  }
  else if(i %in% factor_index) {
    plot(vote_average ~ movies[[i]], data = movies, main = varNames[i], xlab="", 
         col=c("darkorange","dodgerblue3"))
  } else {
    vote_neg <- movies[[i]][movies$vote_average == "0"]
    vote_pos <- movies[[i]][movies$vote_average == "1"]
    plot(density(vote_neg), main = varNames[i], xlab="", col="dodgerblue3", 
         xlim = c(0,quantile(movies[[i]], probs=0.995)))
    lines(density(vote_pos), col="darkorange")
    legend("topright", lty = 1, legend = c("No", "Yes"), col=c("darkorange","dodgerblue3"), cex = 0.7)
  }
}
par(op)


### Preparation of training and test sample for modelling 

set.seed(2000)

`%notin%` <- Negate(`%in%`)
for (i in 1:varCount) {
  if (i %notin% factor_index) {
    movies[[i]] <- scale(movies[[i]])
  }
}

random_index <- sample(1:nrow(movies), floor(0.5*nrow(movies)))
train <- movies[random_index, ]
test <- movies[-random_index, ]

## Evaluation of different models

### k Nearest Neighbor Model

fitControl <- trainControl(method = "cv", number = 5)

k <- c(1:20)
knn_train <- train(vote_average ~ .,
                   data = train, 
                   method = "knn", 
                   trControl = fitControl, 
                   tuneGrid = data.frame(k = k))

print(knn_train )

### Naive Bayes Model

nb_train <- train(vote_average ~ .,
                  data = train,
                  method = "naive_bayes",
                  trControl = fitControl)

print(nb_train)

### Logistic Regression Model

glm_train <- train(vote_average ~ .,
                   data = train,
                   trControl = fitControl,
                   method = "glm")

summary(glm_train)

### Stepwise Logistic Regression Model

s_glm_train <- train(vote_average ~ .,
                     data = train,
                     trControl = fitControl,
                     method = "glmStepAIC")

summary(s_glm_train)

### Classification Tree Model

rpart_train_full <- rpart(vote_average ~ ., 
                          data = train, 
                          control = list(cp = 0))

rsq.rpart(rpart_train_full)

rpart_train_prunned <- prune(rpart_train_full, 
                             cp = 5.4462e-03 )
rpart.plot(rpart_train_prunned)

rpart_train <- train(vote_average ~ .,
                     data = train,
                     method = "rpart",
                     cp = 5.4462e-03,
                     trControl = fitControl)

### Random Forests Model

rf_train <- train(vote_average ~ .,
                  data = train,
                  method = "rf",
                  tuneGrid = data.frame(mtry = 3),
                  trControl = fitControl)

print(rf_train)

### Importance of variables for different models

#### k Nearest Neighbor Model:  
varImp(knn_train)

#### Naive Bayes Model:  
varImp(nb_train)

#### Logistic Regression Model:  
varImp(glm_train)

#### Stepwise Logistic Regression Model:  
varImp(s_glm_train)

####Classification Tree Model:  
varImp(rpart_train)

#### Random Forest Model:
varImp(rf_train)

## Evaluation of results
knn_test <- confusionMatrix(predict(knn_train, test), test$vote_average)
nb_test <- confusionMatrix(predict(nb_train, test), test$vote_average)
glm_test <- confusionMatrix(predict(glm_train, test), test$vote_average)
s_glm_test <- confusionMatrix(predict(s_glm_train, test), test$vote_average)
rpart_test <- confusionMatrix(predict(rpart_train, test), test$vote_average)
rf_test <- confusionMatrix(predict(rf_train, test), test$vote_average)

x <- round(c(knn_test$overall[1],knn_test$byClass[6],knn_test$byClass[5],
             nb_test$overall[1],nb_test$byClass[6],nb_test$byClass[5],
             glm_test$overall[1],glm_test$byClass[6],glm_test$byClass[5],
             s_glm_test$overall[1],s_glm_test$byClass[6],s_glm_test$byClass[5],
             rpart_test$overall[1],rpart_test$byClass[6],rpart_test$byClass[5],
             rf_test$overall[1],rf_test$byClass[6],rf_test$byClass[5])*100,2)
results <- matrix(x, byrow=TRUE, ncol=3)
colnames(results) <- c("Accuracy", "Recall", "Precision")
rownames(results) <- c("k Nearest Neighbor Model","Naive Bayes Model",
                       "Logistic Regression Model","Stepwise Logistic Regression Model",
                       "Classification Tree Model","Random Forest Model")
results <- as.table(results)

results %>%
  kbl() %>%
  kable_material(c("striped", "hover"))
