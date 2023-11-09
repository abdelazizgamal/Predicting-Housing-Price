library(dplyr)
library(tidyr)
library(purrr)
library(corrplot)
library(lares)
library(mlbench)
library(caret)
library(ConfusionTableR)
library(CatEncoders)
library(hash)
library(scales)
library(ggplot2)
library(kernlab)
library(DataExplorer)
library(forcats);library(corrplot);
setwd("D:/faculty/4.2/Distributed Computing _23/Project_Data")
#read data and convert strings to factors
train_df <-read.csv("train.csv",sep=",", stringsAsFactors = TRUE)
#printing out the first 3 entries of data
head(train_df,3)
#show info of data
str(train_df)
p<-plot_missing(train_df)
#Summarize the numeric values and the structure of the data.
summary(select_if(train_df,is.numeric))

#show graph of the target variable
ggplot(train_df, aes(x=SalePrice)) + geom_histogram(col = 'white') + theme_light() +scale_x_continuous(labels = comma)

#check duplicated rows
cat("The number of duplicated rows are", nrow(train_df) - nrow(unique(train_df)))

#show count of missing values in each column
colSums(sapply(select_if(train_df,is.numeric), is.na))#for numeric columns
colSums(sapply(select_if(train_df,is.factor), is.na))#for categorical columns

#missing data percentage of the whole data
sum(is.na(train_df)) / (nrow(train_df) *ncol(train_df))

#drop ID column as its not necessary
train_df$Id <- NULL

#drop columns with missing values > 300
train_df <- train_df[,colSums(is.na(train_df)) < 300]


#trasform important (categorical columns) to factors
cat_car <- c('Condition1','BedroomAbvGr', 'HalfBath', 'KitchenAbvGr','BsmtFullBath', 'BsmtHalfBath', 'MSSubClass')
train_df[,(cat_car)] <- lapply(train_df[,(cat_car)],factor)


#get mode of categorical data
mode <- function(x, na.rm = FALSE) {
  
  if(na.rm){ #if na.rm is TRUE, remove NA values from input x
    x = x[!is.na(x)]
  }
  
  val <- unique(x)
  return(val[which.max(tabulate(match(x, val)))])
}

#replace NAs with either mean or mode.
preprocessing1 <- function(data){
  data<-data %>% mutate_if(is.numeric, ~replace_na(.,round(mean(., na.rm = TRUE))))
  data <-data %>%  mutate_if(is.factor, ~replace_na(.x, mode(.x, na.rm = TRUE))%>% as.factor)
  return(data)
}
#apply encoding on cat variables
preprocessing2 <- function(data){
  
  for( i in cat_var){
    data[[i]] <- transform(h[[i]], data[[i]])
  }
  return(data)
}

train_df <-preprocessing1(train_df)

#split data to cats and numerics
cat_var <- names(train_df)[which(sapply(train_df, is.factor))]
numeric_var <- names(train_df)[which(sapply(train_df, is.numeric))]
train_cat <- train_df[, cat_var]
train_cont <- train_df[ , numeric_var]

# show correlations great than |0.3|
correlations <- cor(na.omit(train_cont))
row_indic <- apply(correlations, 1, function(x) sum(x > 0.2 | x < -0.2) > 1)
correlations<- correlations[row_indic ,row_indic ]
corrplot(correlations, method="square")
#select features that is below cutoff correlation and remove it
n <- row_indic[which(sapply(row_indic,isFALSE))]
train_cont[,names(n)]<- list(NULL)
e<-names(n)

#drop target column 
trainSalePrice <- train_cont$SalePrice
train_cont$SalePrice <- NULL

#SCALING 
X_train_scaled = scale(train_cont)

#encoding
h <- hash()
for( i in cat_var){
  h[[i]] <- LabelEncoder.fit(train_cat[[i]])
}
train_cat <-preprocessing2(train_cat)


train_data = cbind(X_train_scaled,train_cat)
#################training models#################################
## SVM model
library(e1071)
sigDist <- sigest(trainSalePrice ~ ., data = train_data, frac = 1)
svmTuneGrid <- data.frame(sigma = as.vector(sigDist)[1], C = 2^(-2:7))
#sigma =  0.00476585 and C = 2
# fit support vector machine (SVM) model
modSVM <- caret::train(x = train_data,
                       y = trainSalePrice,
                       method = "svmRadial",
                       tuneGrid = svmTuneGrid,
                       trControl = trainControl(method = "boot", number = 50))
modSVM

featureImp_SVM <- varImp(modSVM)
ggplot(featureImp_SVM, mapping = NULL,
       
       top = dim(featureImp_SVM$importance)[1]-(dim(featureImp_SVM$importance)[1]-25), environment = NULL)+
  xlab("Feature")+
  ylab("Importace")+
  theme(text = element_text(size=9))

train_pred_SVM <- predict(modSVM, train_data)
plot(train_pred_SVM,                                
     trainSalePrice,
     xlab = "Predicted Values",
     ylab = "Observed Values")
abline(a = 0,                                        # Add straight line
       b = 1,
       col = "blue",
       lwd = 2)


#randomForest model
library(randomForest)
forest_model <- randomForest(trainSalePrice~.,
                            data = train_data)

train_pred_forest <- predict(forest_model,train_data)
plot(train_pred_forest,                                # Draw plot using Base R
     trainSalePrice,
     xlab = "Predicted Values",
     ylab = "Observed Values")
abline(a = 0,                                        # Add straight line
       b = 1,
       col = "blue",
       lwd = 2)

#generalized linear models
glm_model <- glm(trainSalePrice ~ ., data = train_data, family = "gaussian")
train_pred_glm <- predict(glm_model,train_data)
plot(train_pred_glm,                                # Draw plot using Base R
     trainSalePrice,
     xlab = "Predicted Values",
     ylab = "Observed Values")
abline(a = 0,                                        # Add straight line
       b = 1,
       col = "blue",
       lwd = 2)

# Evaluation RMSE function
MSE1 = mean((train_pred_SVM-trainSalePrice)^2)
paste("SVM MSE: ",MSE1)
MSE2 = mean((train_pred_forest-trainSalePrice)^2)
paste("forest MSE: ",MSE2)
MSE3 = mean((train_pred_glm-trainSalePrice)^2)
paste("glm MSE: ",MSE3)



data <- data.frame(mse = c(MSE1,MSE2,MSE3 ), Model = c("SVM","RandomForest","GLM"),
                   kaggle_scores = c(0.13524,0.14583,0.17299))

lowest_mse_index <- which.min(data$mse)
lowest_kaggle_index <- which.min(data$kaggle_scores)

mse_colors <- rep("red", nrow(data))
mse_colors[lowest_mse_index] <- "green"
  
kaggle_colors <- rep("red", nrow(data))
kaggle_colors[lowest_kaggle_index] <- "green"
## Bar plot for MSE
mse_plot <- ggplot(data, aes(x = Model, y = mse)) +
  geom_bar(stat = "identity", fill = mse_colors) +
  labs(title = "Mean Squared Error (MSE) Comparison", x = "Models", y = "MSE") +
  theme_minimal() +
  theme(text = element_text(size = 14), plot.title = element_text(size = 18),
        axis.title = element_text(size = 16), axis.text = element_text(size = 18),
        axis.text.x = element_text(angle = 45, vjust = 0.5))

mse_plot <- mse_plot +
  geom_text(aes(label = sprintf("%.2f", mse), y = mse), vjust = -0.5, color = "black", size = 4)

print(mse_plot)
## Bar plot for Kaggle scores
kaggle_plot <- ggplot(data, aes(x = Model, y = kaggle_scores)) +
  geom_bar(stat = "identity", fill = kaggle_colors) +
  labs(title = "Kaggle Scores Comparison", x = "Models", y = "Kaggle Score") +
  theme_minimal() +
  theme(text = element_text(size = 14), plot.title = element_text(size = 18),
        axis.title = element_text(size = 16), axis.text = element_text(size = 18),
        axis.text.x = element_text(angle = 45, vjust = 0.5))

kaggle_plot <- kaggle_plot +
  geom_text(aes(label = sprintf("%.5f", kaggle_scores), y = kaggle_scores), vjust = -0.5, color = "black", size = 4)

print(kaggle_plot)


####################Test#############################
test_df=read.csv("test.csv")
IDs <- test_df$Id

#get only the featuers we trained on
library(stringr)
features <-c()
for(i in names(test_df)){
  
  features <- append(features, sum(str_detect(names(train_data), i)) > 0)
}
test_df <- test_df[,features]

test_df[,(cat_var)] <- lapply(test_df[,(cat_var)],as.factor)

#apply prepocessing 

test_df <-preprocessing1(test_df)

#split data
test_cat <- test_df[, cat_var]
numeric_var <- names(test_df)[which(sapply(test_df, is.numeric))]
test_cont <- test_df[ , numeric_var]

#applay the same scaling to train data
X_test_scaled = scale(test_cont, center=attr(X_train_scaled, "scaled:center"), 
                      scale=attr(X_train_scaled, "scaled:scale"))
#apply encoding
test_cat <-preprocessing2(test_cat)
#replace unknown values in encoding with ( 0 )
test_cat <-test_cat %>%  mutate(across(everything(), ~replace_na(.x,0)))

colSums(sapply(test_cat, is.na))
Test_Data <- cbind(X_test_scaled,test_cat)

#predict test_data
svm_pred_Te <- predict(modSVM,newdata = Test_Data)

#make CSV file
df <- data.frame(Id = IDs,
                 SalePrice = svm_pred_Te)
write.csv(df,"predictons.csv", row.names = FALSE)
