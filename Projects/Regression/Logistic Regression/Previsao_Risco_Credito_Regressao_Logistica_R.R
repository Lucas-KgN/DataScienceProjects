# Previsão de Risco de Crédito utilizando a linguagem R

# Através da linguagem R será aplicada a Regressão Logística utilizando uma base de dados com informações de clientes que está na pasta data.

# O objetivo está em criar um modelo que realizará a classificação de um novo cliente, determinando se este pode receber crédito ou não. 


---------------------------------------##---------------------------------------##---------------------------------------------------------##--------------------
# Início do projeto


# Definindo o diretório do trabalho
setwd("D:/UTFPR/Cursos/Machine_Learning/Machine_Learning_Regressao_Parte_2/Pratica_e_Definicoes/R/Feito")
getwd()


# Instalando os Pacotes a serem utilizados, só precisa instalar uma vez
install.packages('ROCR') # Permite criar uma curva para avaliar a performance do modelo
install.packages('e1071', dependencies=TRUE) # Possui funções e algoritmos a serem utilizados
install.packages('caret', dependencies=TRUE) # Permite criar modelos em ML em R

# Carregando os Pacotes instalados
library(caret)
library(ROCR) 
library(e1071) 


# Carregando o dataset em um dataframe
credito_dataset <- read.csv('D:/UTFPR/Cursos/Machine_Learning/Machine_Learning_Regressao_Parte_2/Pratica_e_Definicoes/R/dados/credit_dataset_final.csv', header = TRUE, sep = ',')
head(credito_dataset) # Carrega o dataset
summary(credito_dataset) # Resumo estatistico de cada uma das variáveis
str(credito_dataset) # Tipo das variáveis
View(credito_dataset) # Mostra os dados em formato de tabela


####------------------------- Pré-processamento nos dados ----------------------------------------####

# Pelo fato das variáveis terem sido interpretadas como inteiras é necessário reformular seus tipos. 

# O que for categoria é convertido como fator e o que for valor numérico é normalizado


# Transformando variáveis em fatores através de uma função
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}


# Normalizando os dados por meio de uma função
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center = T, scale = T)
  }
  return(df)
}


# Normalizando as variáveis
numeric.vars <- c('credit.duration.months', 'age', 'credit.amount')
credito_dataset_scaled <- scale.features(credito_dataset, numeric.vars) # Enviando para a função de normalização a base de dados e as variáveis numéricas

# Variáveis do tipo fator
categorical.vars <- c('credit.rating', 'account.balance', 'previous.credit.payment.status',
                      'credit.purpose', 'savings', 'employment.duration', 'installment.rate',
                      'marital.status', 'guarantor', 'residence.duration', 'current.assets',
                      'other.credits', 'apartment.type', 'bank.credits', 'occupation', 
                      'dependents', 'telephone', 'foreign.worker')

credito_dataset_final <- to.factors(df = credito_dataset_scaled, variables = categorical.vars)

# Demonstração da transformação na base de dados
str(credito_dataset_final)
View(credito_dataset_final)


# Preparação dos dados em treino e teste

# Será 60% dos dados para treino e 40% para teste
indexes <- sample(1:nrow(credito_dataset_final), size=0.6 * nrow(credito_dataset_final))
train.data <- credito_dataset_final[indexes,]
test.data <- credito_dataset_final[-indexes,]

# Exibindo o tipo das bases já separadas
class(train.data)
class(test.data)


# Separando os atributos e as classes, ou seja, as variáveis independentes das dependentes
test.feature.vars <- test.data[,-1]
test.class.var <- test.data[,1]

# Exibindo os tipos das bases organizadas
class(test.feature.vars)
class(test.class.var)



####-------------------------------- Criando o Modelo de Regressão Logística ---------------------------------------------####

# Criando o modelo
formula.init <- 'credit.rating ~ .' # A esquerda do sinal TIO esta a variável target, a direita estão as variáveis explanatorias
formula.init <- as.formula(formula.init)
modelo_v1 <- glm(formula = formula.init, data = train.data, family = 'binomial') # O que determina o modelo é o family. No caso binomial porque a saída é se a entrada é ou não aquela classe

# Demonstrando detalhes do modelo criado
summary(modelo_v1)

# Realizando Previsões
previsoes <- predict(modelo_v1, test.data, type='response')
previsoes <- round(previsoes) # Arredonda os valores para 0 ou 1
View(previsoes)

# Utilizando a Confusion Matrix será possível visualizar as métricas obtidas
confusionMatrix(table(data = previsoes, reference = test.class.var), positive = '1')

# A acurácia obtida foi de 73.75% e o valor-p foi abaixo de 0.05. Portando pode-se concluir que o modelo apresentou um resultado bom, porém pode ser melhorado.


####----------------------------------- Feature Selection Buscando a melhoria do modelo ----------------------------------------------####

# Feature Selection
formula <- 'credit.rating ~ .'
formula <- as.formula(formula)
control <- trainControl(method = 'repeatedcv', number = 10, repeats = 2) # Essa função repete o treinamento do modelo pelo numero de vezes escolhida e traz as variáveis mais relevantes.
model <- train(formula, data = train.data, method = 'glm', trControl = control)
importance <- varImp(model, scale = FALSE) # Essa função traz a importância das variáveis.
  

# Plotando um gráfico com as variáveis
plot(importance)
  
# Verificou-se que as variáveis account.balance, credit.purpose, previous.credit.payment.status, savings, credit.duration.months e installment.rate são relevantes para o resultado final e portanto serão utilizadas.
  

# Portanto um novo modelo será construído utilizando somente essas variáveis preditivas
formula.new <- 'credit.rating ~ account.balance + credit.purpose + previous.credit.payment.status + savings + credit.duration.months + installment.rate'
formula.new <- as.formula(formula.new)
modelo_v2 <- glm(formula = formula.new, data = train.data, family = 'binomial')


# Exibindo as métricas obtidas
summary(modelo_v2)


# Prevendo e avaliando o novo modelo
previsoes_new <- predict(modelo_v2, test.data, type = 'response')
previsoes_new <- round(previsoes_new)


# Revelando a acurácia obtida
confusionMatrix(table(data = previsoes_new, reference = test.class.var), positive = '1')
  
# A acurácia obtida foi muito próxima da anterior, além de ter sido menor. Porém não significa que seja um problema, pois o modelo anterior poderia ser tendencioso por utilizar todas as variáveis. 


####---------------------------------------Avaliando a performance do modelo---------------------------------------------------------------------------####

# Através da Curva ROC é que o modelo será avaliado ao inserir dados novos de teste. 

# Plot do modelo com melhor acurácia
modelo_final <- modelo_v2
previsoes <- predict(modelo_final, test.feature.vars, type = "response")
previsoes_finais <- prediction(previsoes, test.class.var)


# Função para Plot ROC 
plot.roc.curve <- function(predictions, title.text){
  perf <- performance(predictions, "tpr", "fpr")
  plot(perf,col = "black",lty = 1, lwd = 2,
       main = title.text, cex.main = 0.6, cex.lab = 0.8,xaxs = "i", yaxs = "i")
  abline(0,1, col = "red")
  auc <- performance(predictions,"auc")
  auc <- unlist(slot(auc, "y.values"))
  auc <- round(auc,2)
  legend(0.4,0.4,legend = c(paste0("AUC: ",auc)), cex = 0.6, bty = "n", box.col = "white")
}


# Plot
par(mfrow = c(1, 2))
plot.roc.curve(previsoes_finais, title.text = "Curva ROC")


# Portanto é possível afirmar que a Curva ROC do modelo V2 está acima da linha central, ou seja, ele obteve um desempenho acima de 50%, que no caso foi de 77% de acurácia.



####-------------------------------------------------Realizando novas predições com o modelo-----------------------------------------------------####

# Novos dados serão apresentados ao modelo e ele terá que realizar previsões. 


# Novos dados simbolizando novos 6 clientes
account.balance <- c(1, 3, 3, 2, 1, 2)
credit.purpose <- c(4, 2, 3, 2, 3, 4)
previous.credit.payment.status <- c(3, 3, 2, 2, 3, 2)
savings <- c(2, 3, 2, 3, 2, 3)
credit.duration.months <- c(15, 12, 8, 6, 10, 8)
installment.rate <- c(3, 2, 4, 1, 2, 1)


# Criando um dataframe
novo_dataset <- data.frame(account.balance,
                           credit.purpose,
                           previous.credit.payment.status,
                           savings,
                           credit.duration.months,
                           installment.rate)
class(novo_dataset)
# Exibindo o Dataframe com os novos clientes
View(novo_dataset)


# Pré-processamento dos novos dados
# Separando as variáveis explanatórias numéricas e categóricas
new.numeric.vars <- c("credit.duration.months")
new.categorical.vars <- c('account.balance', 'previous.credit.payment.status', 
                          'credit.purpose', 'savings', 'installment.rate')

# Aplica as transformações
novo_dataset_final <- to.factors(df = novo_dataset, variables = new.categorical.vars)
str(novo_dataset_final)

novo_dataset_final <- scale.features(novo_dataset_final, new.numeric.vars)
str(novo_dataset_final)

View(novo_dataset_final)


# Previsões do modelo
previsao_novo_cliente <- predict(modelo_final, newdata = novo_dataset_final, type = "response")
round(previsao_novo_cliente)

# Portanto segundo o modelo somente o cliente de índice 1 não receberia o crédito. 














