#########################################################
# Análisis tasa de cancelación en el ámbito de clientes #
######################### CHURN #########################
#########################################################

library(ggplot2)
library(caret)
library(rpart)
library(C50)
library(rattle)
library(party)
library(partykit)
library(randomForest)
library(ROCR)
library(reshape2)
library(car)
library(corrplot)
library(e1071)
library(psych)

rm(list=ls()) 
setwd("...")

mydata<-read.csv("churn.csv",header = TRUE)



# Analisis de la información. # intl-> international

colnames(mydata)
colnames(mydata)<-(c("State","Accountlength","AreaCode","PhoneNumber","IntlPlan",
                     "VoiceMail","NumVoiceMail","TotDayMin","TotDayCall","TotDayCharge","TotEveMin",
                     "TotEveCall","TotEveCharge","TotNigthMin","TotNightCall","TotNightCharge",
                     "TotintlMin","TotintlCall","TotintlCharge","NumCustServCall","Churn"))
colnames(mydata)

# Cambio de las variables a su forma numerica.

mydata$Churn<-as.integer(mydata$Churn)
mydata$IntlPlan<-as.integer(mydata$IntlPlan)
mydata$VoiceMail<-as.integer(mydata$VoiceMail)

mydata$Churn[mydata$Churn=="1"]<-0
mydata$Churn[mydata$Churn=="2"]<-1

mydata$IntlPlan[mydata$IntlPlan=="1"]<-0
mydata$IntlPlan[mydata$IntlPlan=="2"]<-1

mydata$VoiceMail[mydata$VoiceMail=="1"]<-0
mydata$VoiceMail[mydata$VoiceMail=="2"]<-1

# Remover las variables no necesarias.

mydata$State<-NULL
mydata$AreaCode<-NULL
mydata$PhoneNumber<-NULL


# Eliminar las observaciones faltantes del conjunto de datos.

na.omit(mydata)


summary(mydata)

# Calculo de la desviación estandar.

sapply(mydata, sd)


cormatrix<-round(cor(mydata),digits = 2)
cormatrix

# Mapa de correlación de las variables.
pairs.panels(cormatrix, col="red",cex.labels=1.3,cex.cor = 5)
corrplot(cormatrix,method="number",title = "Correlacion variables clientes.")


qplot(x=Var1,y=Var2,data = melt(cor(mydata,use="p")),fill=value,geom = "tile")+
      scale_fill_gradient2(limits=c(-1,1))

plot.new()
plot(mydata$Churn~mydata$TotDayMin)
title("Grafica de dispersión.")

plot.new()
hist(mydata$TotDayMin)

plot.new()
boxplot(mydata$TotDayMin)

# Empleo de la libreria ggplot.
ggplot(mydata,aes(x=mydata$TotDayMin,y=mydata$TotDayMin))+geom_point(size = 3)

ggplot(mydata,aes(x=mydata$TotDayMin,y=mydata$TotDayMin,colour=mydata$Churn))+geom_point(size = 3)

ggplot(mydata,aes(x=mydata$TotDayMin))+geom_histogram(binwidth = 1,fill="blue",colour="black")

# Division del conjunto de datos en entrenamiento y prueba.
set.seed(20)
ind<-sample(2,nrow(mydata),replace = TRUE, prob = c(0.7,0.3))
trainData<-mydata[ind==1,]
testData<-mydata[ind==2,]

#################################################################################
# Modelo 1. Modelo de Regresión Logistico.
#################################################################################

# Menor valor AIC indicta un mejor modelo

# Eliminación hacia adelante (Forward Elimination)

forwardtest<-step(glm(Churn~1,data=trainData),direction="forward",
                  scope = ~Accountlength+IntlPlan+VoiceMail+NumVoiceMail+TotDayMin+TotDayCall+
                    TotDayCharge+TotEveMin+TotEveCall+TotEveCharge+TotNigthMin+TotNightCall+
                    TotNightCharge+TotintlMin+TotintlCall+TotintlCharge+NumCustServCall)

capture.output(forwardtest,file="test2b.doc")

# Resultados del modelo de regresión logistica.

mylogit<-glm(Churn~ TotDayCharge+TotNightCall+NumVoiceMail+TotEveCall+TotDayCall+TotNigthMin+
               TotintlCharge+TotEveMin,data = trainData,family = "binomial")
summary(mylogit)

# Evaluación del ajuste del modelo y desempeño.
influenceIndexPlot(mylogit,vars=c("Cook","hat"),id.n = 3)

# CIs using profiled log-likehood.
confint(mylogit)

# CIs using standar errors.
confint.default(mylogit)

# Escalamiento de los coeficientes.

exp(mylogit$coefficients)
exp(confint(mylogit))

#odds rations only
exp(coef(mylogit))

# odds rations and 95% CI
exp(cbind(OR=coef(mylogit),confint(mylogit)))


#################################################################################
# Modelo 2. Support Vector Machines.
#################################################################################

SVMModel<-svm(Churn~., data=trainData,gamma=0.1,cost=1)
summary(SVMModel)

#################################################################################
# Modelo 3. Random Forest Model.
#################################################################################

RandomForestModel<-randomForest(Churn~.,data=trainData,ntree=500,ntry=5,inportance=TRUE)
summary(RandomForestModel)
print(RandomForestModel)
importance(RandomForestModel)

plot.new()
varImpPlot(RandomForestModel,type=2,pch=19,col=1,cex=1.0,main="Random Forest")
abline(v=30,col="blue")


#################################################################################
# Obtención de información. Construcción de un arbol de desiciones usando c5.0.
#################################################################################

# La variable de desicion tiene que ser convertida 
# en factor para que el algoritmo C50 funcione correctamente.

mydata$Churn<-as.factor(mydata$Churn)

# Algoritmo del arbol de desición.
c50_tree_result<-C5.0(Churn~.,data=mydata)
summary(c50_tree_result)

C5imp(c50_tree_result,metric = "usage")
C5imp(c50_tree_result,metric = "splits")

############################################
# Visualización de las reglas de desición.
############################################

c50_rule_result<-C5.0(Churn~.,data = mydata,rules=TRUE)
summary(c50_rule_result)

# Evaluación del desempeño del modelo.
LogisticModel<-predict(mylogit,testData,type = "response")
SVMResult<-predict(SVMModel,testData,type="reponse")
RFResult<-predict(RandomForestModel,testData,type="response")

# Definir los resultados como columnas del conjuto de datos.
testData$YHat1<-predict(mylogit,testData,type="response")
testData$YHat2<-predict(SVMModel,testData,type="response")
testData$YHat3<-predict(RandomForestModel,testData,type="response")

# Parametros de control.
Predict<-function(t)ifelse(LogisticModel >t,1,0)
Predict2<-function(t)ifelse(SVMResult >t,1,0)
Predict3<-function(t)ifelse(RFResult >t,1,0)


confusionMatrix(Predict(0.5),testData$Churn)
confusionMatrix(Predict2(0.5),testData$Churn)
confusionMatrix(Predict3(0.5),testData$Churn)


############################################
# ROC for unpruned model.
############################################

pred1 <- prediction(testData$YHat1,testData$Churn)
pred2 <- prediction(testData$YHat2,testData$Churn)
pred3 <- prediction(testData$YHat3,testData$Churn)

perf<-performance(pred1,"tpr","fpr")
perf2<-performance(pred2,"tpr","fpr")
perf3<-performance(pred3,"tpr","fpr")

plot.new()
plot(perf,col="green",lwd=2.5)
plot(perf2,add=TRUE,col="blue",lwd=2.5)
plot(perf3,add=TRUE,col="orange",lwd=2.5)
abline(0,1,col="red",lwd=2.5,lty=2)
title("ROC Curve")
legend(0.8,0.4,c("Logistic","SVM","RF"),
       lty=c(1,1,1),
       lwd = c(1.4,1.4,1.4),col = c("green","blue","orange","yellow"))


# AUC calculation metrics

fit.auc1<-performance(pred1,"auc")
fit.auc2<-performance(pred2,"auc")
fit.auc3<-performance(pred3,"auc")


fit.auc1
fit.auc2
fit.auc3

########## Guarda el Modelo ##########

save(RandomForestModel,file="ChurnModel.rda")























