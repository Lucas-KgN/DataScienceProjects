# Construindo um modelo de decisão para risco de crédito ao cliente

# Os dados estão disponíveis na pasta dados com o nome de credito.csv


# Selecionando a pasta do projeto
getwd()
setwd('D:/UTFPR/Cursos/Machine_Learning/Decision_Tree_Random_Forest_e_Metodos_Ensemble_Parte_1/Pratica/R/Feito')

# Lendo e visualizando o dataset
credit_db <- read.csv('D:/UTFPR/Cursos/Machine_Learning/Decision_Tree_Random_Forest_e_Metodos_Ensemble_Parte_1/Pratica/R/dados/credito.csv')
str(credit_db)
View(credit_db)

# A base apresenta algumas variáveis que possuem informações sobre clientes que pediram empréstimo e a variável target é a default que determina se o empresto foi pago ou não


# Verificando os saldos dos clientes na conta corrente e poupança
table(credit_db$checking_balance)
table(credit_db$savings_balance)

# Verificando outras características, duração do emprestimo e valor
summary(credit_db$months_loan_duration)
summary(credit_db$amount)

# Verificando a variável target
table(credit_db$default)

# Portanto, 700 pessoas não pagaram o emprestimo e somente 300 pagaram

# Separando a base de dados em treino e teste
set.seed(123)
train_sample <- sample(1000,900)

# Split nos dataframes
credit_train <- credit_db[train_sample,]
credit_test <- credit_db[-train_sample,]

# Verificando a proporção da variável target
prop.table(table(credit_train$default))
prop.table(table(credit_test$default))

# Construindo um modelo com o algoritmo C5.0
install.packages("C50")
library(C50)
?C5.0

# Convertendo a variável target para fator
credit_train$default<-as.factor(credit_train$default)
str(credit_train$default)

# Modelo sendo criado e treinado
credit_model <- C5.0(credit_train[-17], credit_train$default)
credit_model

# Informações sobre a árvore
summary(credit_model)

# Portanto, a árvore possui 69 pontos de decisão. Ela recebe 900 casos de  entrada com  17 atributos. 
# O modelo acertou 89%.

# Realizando previsões com o modelo
credit_pred <- predict(credit_model, credit_test)

# Avaliado o resultado utilizando matriz de confusão
install.packages('gmodels')
library(gmodels)

# Criando a matriz de confusão
CrossTable(credit_test$default,
           credit_pred,
           prop.chisq = FALSE,
           prop.c = FALSE,
           prop.r = FALSE,
           dnn = c('Esperado', 'Previsto'))



# Melhorando os resultados obtidos com 10 tentativas
credit_boost10 <- C5.0(credit_train[-17], credit_train$default, trials = 10)
summary(credit_boost10)

# Este modelo tem menos nós que a árvore anterior, ou seja, ele aprende melhor com menos nós.

# Score do modelo
credit_boost_pred10 <- predict(credit_boost10, credit_test)

# Criando a matriz de confusão
CrossTable(credit_test$default,
           credit_boost_pred10,
           prop.chisq = FALSE,
           prop.c = FALSE,
           prop.r = FALSE,
           dnn = c('Esperado', 'Previsto'))



# Melhorando os resultados novamente dando pesos aos erros

# Criando uma matriz de dimensões de custo
matrix_dimensions <- list(c("no", "yes"), c("no", "yes"))
names(matrix_dimensions) <- c("Esperado", "Observado")
matrix_dimensions

# Construindo a matriz penalizando os erros
error_cost <- matrix(c(0, 1, 4, 0), nrow = 2, dimnames = matrix_dimensions)
error_cost

# Aplicando a matriz a árvore
credit_cost <- C5.0(credit_train[-17], credit_train$default, costs = error_cost)

# Score do modelo
credit_cost_pred <- predict(credit_cost, credit_test)

# Confusion Matrix
CrossTable(credit_test$default, 
           credit_cost_pred,
           prop.chisq = FALSE, 
           prop.c = FALSE, 
           prop.r = FALSE,
           dnn = c('Observado', 'Previsto'))

# Portanto, a árvore criada acima não obteve melhor resultado se comparada aos modelos anteriores.  