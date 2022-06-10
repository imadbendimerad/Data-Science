install.packages("caret")
install.packages("doParallel")
install.packages("parallel")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("rmarkdown")


#DM Introduction en data-science

rm(list=ls())


##I Jeu de données
cardio <- read.csv("processed.cleveland.data", header = FALSE, na.strings = '?')
names(cardio) <- c( "age", "sex", "cp", "trestbps", "chol",
                    "fbs", "restecg", "thalach","exang",
                    "oldpeak","slope", "ca", "thal", "status")
###1 
str(cardio)

###2 Transformation des variables explicatives qualitatives 
?factor

cardio$sex <- factor( cardio$sex,levels = c(0, 1 ),
                      labels = c( "Female","Male") )

cardio$cp <- factor( cardio$cp,levels = c(1, 2, 3, 4 ),
                     labels = c( "typical angina","atypical angina","non-anginal pain","asymptomatic") )

cardio$fbs <- factor( cardio$fbs,levels = c(0, 1 ),
                      labels = c( "fasting blood sugar < 120 mg/dl","fasting blood sugar > 120 mg/dl") )

cardio$thal <- factor( cardio$thal,levels = c(3,6,7),
                       labels = c( "normal","fixed defect","reversable defect") )

cardio$slope <- factor( cardio$slope,levels = c(1,2,3),
                        labels = c( "upsloping","flat","downsloping") )

cardio$exang <- factor( cardio$exang,levels = c(0,1),
                        labels = c( "no","yes") )

cardio$restecg <- factor( cardio$restecg,levels = c(0,1,2),
                          labels = c( "normal","ST-T","LVH") )
cardio$status <- factor( cardio$status,levels = c(0, 1,2,3,4 ),
                         labels = c("0", "1","2","3","4" ) )

str(cardio) 

###3
levels(cardio$status)

###4 Transformation en binaire


cardio$status <- factor( cardio$status,levels = c("0","1","2","3","4"),
                         labels = c( "0","1","1","1","1") )


summary(cardio)

###5 na_count contiendra le nombre de données manquantes

na_count <- length(which(is.na(cardio)))
print(na_count)

?na.omit

cardio <- na.omit(cardio)

## Modelisation

###6

set.seed(1)
train = sample(1:nrow(cardio), round(0.70*nrow(cardio)))
cardio.train = cardio[train,]
cardio.test = cardio[-train,]

###7

## CART sans élagage
require(rpart)
require(rpart.plot)
set.seed(1)
cart.0 <- rpart(status~.,
                data=cardio.train, 
                control=rpart.control(minsplit=5,cp=0, xval=5))

rpart.plot(cart.0)
pred.0 <- predict(cart.0, cardio.test, type ="class")
TMC_sansE <- mean(pred.0!=cardio.test$status)
print(TMC_sansE)


## CART avec élagage
plotcp(cart.0)
cart.0$cptable
which.min(cart.0$cptable[,"xerror"])
cpOptim = cart.0$cptable[which.min(cart.0$cptable[,"xerror"]),"CP"]
cart.pruned <- prune(cart.0, cpOptim)
rpart.plot(cart.pruned)
pred.pruned <- predict(cart.pruned, cardio.test, type ="class")
TMC_avecE <- mean(pred.pruned!=cardio.test$status)
print(TMC_avecE)

print(TMC_sansE)
print(TMC_avecE)

par(mfrow = c(1, 2))
rpart.plot(cart.0)
rpart.plot(cart.pruned)

cart.0$variable.importance
cart.0$variable.importance/sum(cart.0$variable.importance)

cart.pruned$variable.importance
cart.pruned$variable.importance/sum(cart.pruned$variable.importance)


#####


require(caret)
require(doParallel)
require(parallel)

### Random forest 
cl <- makePSOCKcluster(detectCores()-1)
registerDoParallel(cl)
control <- trainControl(method="repeatedcv", number=5, repeats=100)
rfGrid <-  expand.grid(mtry = 1:13)
?train
RFmodel <- train(x = cardio.train[,-14],
                 y = cardio.train$status,
                 method="rf",
                 trControl=control,
                 n.trees=500,
                 tuneGrid = rfGrid,
)

stopCluster(cl)
plot(RFmodel)
pred.rf.caret <- predict(RFmodel, cardio.test)
TMC_RF <- mean(pred.rf.caret!=cardio.test$status) 
print(TMC_RF)

TMC_avecE<TMC_sansE
TMC_RF<TMC_avecE


### ###  #####  #####    ####   #
# # # #  #      #   #   #       #
# # # #  #####  #  #   #        #
#     #  #      #   #   #       #
#     #  #####  #    #   ####   #