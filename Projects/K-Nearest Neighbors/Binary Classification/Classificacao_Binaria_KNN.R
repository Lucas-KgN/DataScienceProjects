# Classificação Binária utilizando o algoritmo KNN 

# O problema consiste em prever o resultado do índice S&P para aquele dia. 

# A base de dados pode ser encontrada no link a seguir: https://rdrr.io/cran/ISLR/man/Smarket.html

# A partir da base de dados fornecida um algoritmo KNN será aplicado para classificar e trazer a previsão esperada.


####---------------------------------Pré-Definições-----------------------------------------------------------------####

# Definindo o diretório
getwd()
setwd('D:/UTFPR/Cursos/Machine_Learning/Classificação_com_k_nearest_neighbours/Pratica_e_Definicoes/R/Gabarito')

# Instalando os pacotes a serem utilizados
install.packages('ISLR')


# Carregando os pacotes a serem utilizados
library(ISLR) # Contém o dataset a ser utilizado
library(caret) # Pacote que possui ferramentas de aprendizado de máquina
library(e1071) # Funções auxiliares


# Definindo o seed
set.seed(300)


####--------------------------------Carregando e explorando o dataset-------------------------------------------------####

# Base de dados
summary(Smarket) # Informações estatísticas das variáveis
str(Smarket) # Informações gerais das variáveis
View(Smarket) # Exibição dos dados


# Split da base de dados em treino e tete

indxTrain <- createDataPartition(y = Smarket$Direction, p=0.75, list=FALSE) # 75% da base para treinamento. Essa função cria um índice.
dados_treino <- Smarket[indxTrain,]
dados_teste <- Smarket[-indxTrain,]

# Demonstrando que foram criados dois dataframes com os dados
class(dados_treino)
class(dados_teste) 


# Verificando a distribuição dos dados originais e as partições feitas
prop.table(table(Smarket$Direction)) * 100 # Essa função retorna a proporção
prop.table(table(dados_treino$Direction)) * 100

# Portanto tanto a base original de dados quanto a base de treino estão bem distribuídas. 


# Correlação entre as variáveis preditoras
descCor <- cor(dados_treino[, names(dados_treino) != 'Direction']) # Exclui a coluna com as respostas alvo
descCor

# Portanto, existem variáveis que possuem um grau de correlação relevante, porém nenhuma delas apontou o problema de multicolinearidade.


####-----------------------------Normalizando os dados----------------------------------------------------------------------------------------####

# Função de normalização utilizando Center que utiliza o valor da média dos atributos e subtrai o valor das linhas por ela e Scale que calcula o desvio padrão e divide o valor da linha por ele.
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center = T, scale = T)
  }
  return(df)
}

# Removendo a variável target dos dados de treino e teste
numeric.vars_treino <- colnames(treinoX <- dados_treino[,names(dados_treino) != 'Direction'])
numeric.vars_teste <- colnames(testeX <- dados_teste[,names(dados_teste) != 'Direction'])


# Aplicando a normalização nas bases de treino e teste
dados_treino_scaled <- scale.features(dados_treino, numeric.vars_treino)
dados_teste_scaled <- scale.features(dados_teste, numeric.vars_teste)


# Exibindo os dados normalizados
View(dados_treino_scaled)
View(dados_teste_scaled)


####-----------------------------------------------Criando e treinando o modelo KNN--------------------------------------------------------####

# Criando o arquivo de controle
ctrl <- trainControl(method = 'repeatedcv', repeats = 5) # Por utilizar validação cruzada buscando o melhor valor de K é que foi preciso criar essa variável.


# Criação e treinamento do modelo 
knn_V1 <- train(Direction ~ .,
                data = dados_treino_scaled,
                method = 'knn',
                trControl = ctrl,
                tuneLength = 20,
                #preProcess = c('center', 'scale') # Caso fosse escolhido realizar a normalização neste momento este seria o comando
                ) 



####--------------------------------------------------Avaliação do modelo-------------------------------------------------------------------------------####

# Exibindo as métricas do modelo
knn_V1


# Gráfico com as métricas
plot(knn_V1)


# Realizando previsões com o modelo criado
knnPredict <- predict(knn_V1, newdata = dados_teste_scaled)


# Matriz de Confusão para avaliar as previsões
confusionMatrix(knnPredict, dados_teste$Direction)

# Portanto o modelo obteve uma acurácia relevante, de 89,42% e um valor-p muito baixo. 

 

####-------------------------------------------Alterando as métricas de controle--------------------------------------------------------------------------####

# Alterações vão ocorrer nas métricas de controle aplicadas ao modelo, buscando um melhor resultado. 


# Novo arquivo de controle
ctrl <- trainControl(method = 'repeatedcv',
                     repeats = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)


# Novo treinamento do modelo
knn_V2 <- train(Direction ~ .,
                data = dados_treino_scaled,
                method = 'knn',
                trControl = ctrl,
                tuneLength = 20,
                metric = 'ROC',
                #preProcess = c('center', 'scale') # Caso fosse escolhido realizar a normalização neste momento este seria o comando
) 


# Exibição do modelo
knn_V2


# Exibindo a relação entre o valor K e a acurácia ROC
plot(knn_V2)


# Fazendo previsões
knnPredict <- predict(knn_V2, newdata = dados_teste_scaled)


# Criando uma matriz de confusão para verificar os resultados obtidos
confusionMatrix(knnPredict, dados_teste$Direction)

# Ao alterar algumas métricas de controle e o método de acurácia utilizada obteve-se um aumento pequeno no valor da acurácia.



####------------------------------------------Gerando previsões com base em novos dados------------------------------------------------------------####

# Preparando os dados de entrada
Year = c(2006, 2007, 2008)
Lag1 = c(1.30, 0.09, -0.654)
Lag2 = c(1.483, -0.198, 0.589)
Lag3 = c(-0.345, 0.029, 0.690)
Lag4 = c(1.398, 0.104, 1.483)
Lag5 = c(0.214, 0.105, 0.589)
Volume = c(1.36890, 1.09876, 1.231233)
Today = c(0.289, -0.497, 1.649)

novos_dados = data.frame(Year, Lag1, Lag2, Lag3, Lag4, Lag5, Volume, Today)
novos_dados
str(novos_dados)
class(novos_dados)


##------------------------Normalizando os dados---------------------------------------##

# Extraindo os nomes das variáveis
nomes_variaveis <- colnames(novos_dados)
nomes_variaveis

# Aplicando a função de normalização
novos_dados_scaled <- scale.features(novos_dados, nomes_variaveis)
novos_dados_scaled
str(novos_dados_scaled)
class(novos_dados_scaled)

# Fazendo previsões
knnPredict <- predict(knn_V2, newdata = novos_dados_scaled)
cat(sprintf("\n Previsão de \"%s\" é \"%s\"\n", novos_dados$Year, knnPredict))

# Portanto o modelo performa bem, além de realizar previsões em novos dados. 


















