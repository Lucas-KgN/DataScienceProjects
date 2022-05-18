# Criando Árvore de Decisão com o pacote rpart para definir se é possível ou não jogar tenis com base nas informações sobre o dia

# O pacote a ser utilizado será o expand.grid que permite criar um Dataframe a partir das combinações de variáveis categóricas

# Configurando o diretório de armazenamento
getwd()
setwd("D:/UTFPR/Cursos/Machine_Learning/Decision_Tree_Random_Forest_e_Metodos_Ensemble_Parte_1/Pratica/R/Feito")

  
# Criando um Dataframe com 4 variáveis
clima <- expand.grid(Tempo = c("Ensolarado", "Nublado", "Chuvoso"),
                     Temperatura = c("Quente", "Ameno", "Frio"),
                     Humidade = c("Alta", "Normal"),
                     Vento = c("Fraco", "Forte"))

# Visualizando o Dataframe criado
View(clima)

# Vetor para selecionar as linhas que serão utilizadas
response <- c(1, 19, 4, 31, 16, 2, 11, 23, 35, 6, 24, 15, 18, 36) 

# Gerando um vetor do tipo fator para a Variável target
play <- as.factor(c("Não Jogar", "Não Jogar", "Não Jogar", "Jogar", "Jogar", "Jogar", "Jogar", "Jogar", "Jogar", "Jogar", "Não Jogar", "Jogar", "Jogar", "Não Jogar")) 

# Dataframe finals
tennis <- data.frame(clima[response,], play) # Selecionando somente as linhas com index apontado pela variável response e adicionando as informações da variável target
View(tennis)

# Carregando o pacote R para a arvore
library(rpart)

# Construindo o modelo
model = rpart(play ~ .,
              data = tennis,
              method = 'class',
              parms = list(split = 'information'),
              control = rpart.control(minsplit = 1))

# Visualizando a arvore
install.packages("rpart.plot")
library(rpart.plot)

prp(model, type = 0, extra = 1, under = TRUE, compress = TRUE)


# Realizando previsões

# Dados
clima <- expand.grid(Tempo = c("Ensolarado", "Nublado", "Chuvoso"),
                     Temperatura = c("Quente", "Ameno", "Frio"),
                     Humidade = c("Alta", "Normal"),
                     Vento = c("Fraco", "Forte"))

# Vetor para selecionar as linhas que serão utilizadas ainda não vistas
response <- c(2, 20, 3, 33, 17, 4, 5) 

# Novos dados
novos_dados <- data.frame(clima[response,])
View(novos_dados)

# Previsões
predict(model, novos_dados)

# O resultado do modelo apresenta alguns erros nas previsões, por isso seria interessante ajusta-lo com novos dados.

# Aplicando o Prunning na árvore
pruned_model <- prune(model, cp=0.2)

# Exibindo o modelo pós poda
prp(pruned_model, type = 0, extra = 1, under = TRUE, compress = TRUE)
