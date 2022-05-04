library(keras)


# Train -------------------------------------------------------------------
train <- readr::read_csv('C:\\Studenci\\Biostatystyka I mgr\\AJ\\randomForrest_3\\train.csv')
train<-train[,-c(1,4,9,11)]

train$Embarked<-as.integer(as.factor(train$Embarked))-1
train$Sex<-as.integer(as.factor(train$Sex))-1
train$Pclass<-as.integer(as.factor(train$Pclass))-1

train <- na.omit(train)

trained<-as.matrix(train)
dimnames(trained)<-NULL
trainy<-trained[,1]
trainx<-trained[,-1]
trainlabel<-to_categorical(trainy)


# Test --------------------------------------------------------------------

test <- readr::read_csv('C:\\Studenci\\Biostatystyka I mgr\\AJ\\randomForrest_3\\test.csv')
test<-test[,-c(1,3,8,10)]
test_id <- readr::read_csv('C:\\Studenci\\Biostatystyka I mgr\\AJ\\randomForrest_3\\test.csv') %>% 
  dplyr::select(PassengerId)


test$Embarked<-as.integer(as.factor(test$Embarked))-1
test$Sex<-as.integer(as.factor(test$Sex))-1
test$Pclass<-as.integer(as.factor(test$Pclass))-1

test <- na.omit(test)

tested<-as.matrix(test)
dimnames(tested)<-NULL
#testy<-tested[,1]
testx<-tested
#testlabel<-to_categorical(testy)

# Model -------------------------------------------------------------------
model <- keras_model_sequential()
model %>%
  layer_dense(units=10,activation = "relu",
              kernel_initializer = "he_normal",input_shape =c(7))%>%
  layer_dense(units=2,activation = "sigmoid")
summary(model)  

model %>%
  compile(loss="binary_crossentropy",
          optimizer="adam",
          metric="accuracy")

history <- model %>%
 fit(trainx,trainlabel,epoch=100,batch_size=20,validation_split=0.2)

train_eva <- model %>%
  evaluate(trainx,trainlabel)

# Tuning ------------------------------------------------------------------

# One more layer with 20 nodes
# Epochs increased to 200

model1 <- keras_model_sequential()

model1 %>%
  layer_dense(units=10,activation = "relu",
              kernel_initializer = "he_normal",input_shape =c(7)) %>%
  layer_dense(units=20, activation = "relu",
              kernel_initializer = "he_normal") %>%
  layer_dense(units=2,activation = "sigmoid")

model1 %>%
  compile(loss="binary_crossentropy",
          optimizer="adam",
          metric="accuracy")

history1 <- model1 %>%
  fit(trainx,trainlabel,epoch=100,batch_size=20,validation_split=0.2)

train_eva1 <- model1 %>%
  evaluate(trainx,trainlabel)

# Prediction --------------------------------------------------------------

y_test_cm <- model1 %>% predict(trainx)

y_test_pred <- ifelse(y_test_cm[,1] > 0.5, 0, 1)

confusionMatrix(factor(trainy), factor(y_test_pred))



test_pred <- model1 %>% predict(testx)

test_pred_bin <- ifelse(test_pred[,1] > 0.5, 0, 1)

confusionMatrix(factor(trainy), factor(y_test_pred))


