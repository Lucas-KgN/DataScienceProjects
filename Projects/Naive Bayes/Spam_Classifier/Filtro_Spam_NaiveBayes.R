# 

# Este projeto consiste na aplicação do algorito Naive Bayes para criar um filtro de mensagem que detecta Spam via SMS.

# A base de dados utilizada pode ser encontrada no link a seguir: //www.dt.fee.unicamp.br/~tiago/smsspamcollection/


# Instalando os pacotes a serem utilizados
install.packages('tm')
install.packages('SnowballC')
install.packages('wordcloud')
install.packages('gmodels')

# Carregando os pacotes
library(tm) # Feito para mineração de textos
library(SnowballC)
library(wordcloud) # Feito para construir uma nuvem de palavras para o processamento
library(e1071) # Contém o algoritmo Naive Bayes
library(gmodels) # Para construir a matrix de confusão


# Carregando a base de dados
dados <- read.csv('D:/UTFPR/Cursos/Machine_Learning/Classificacao_com_Naive_Bayes/Pratica_e_Definicoes/R/dados/sms_spam.csv')

# Examinando a estrutura da base
str(dados)
View(dados)

# A base de dados consiste em um dataframe com 5559 linhas e 2 colunas. Uma coluna armazena o tipo de mensagem e a outra o texto.


####-------------------------------------------------------------- Pré-processamento--------------------------------------------------------------------####

# Convertendo para fator a coluna que determina se é ou não spam
dados$type <- factor(dados$type)


# Examinando a estrutura dos dados novamente
str(dados$type)
table(dados$type)


# Construindo um conjunto de documento de texto, um Corpus
dados_corpus <- VCorpus(VectorSource(dados$text))
View(dados_corpus)

# Examinando a estrutura dos dados
print(dados_corpus)
inspect(dados_corpus[1:2]) # Por ser valores do tipo Corpus pode-se usar o inspect para verificar

# Ajustando a estrutura do Corpus
as.character(dados_corpus[[1]]) # Convertendo para caractere
lapply(dados_corpus[1:2], as.character) # O lappy aplica um mesmo comando a cada linha da base

# Limpeza do Corpus com tm_map()
dados_corpus_clean <- tm_map(dados_corpus, content_transformer(tolower)) # Realiza transformações no Corpus. No caso converteu os caracteres para minusculos.
dados_corpus_clean <- tm_map(dados_corpus_clean, removeNumbers) # Remove os números
dados_corpus_clean <- tm_map(dados_corpus_clean, removeWords, stopwords()) # Remove as stop words
dados_corpus_clean <- tm_map(dados_corpus_clean, removePunctuation) # Remove a pontuação

# Demonstrando a diferença entre o Corpus original e pós limpeza
as.character(dados_corpus[[1]])
as.character(dados_corpus_clean[[1]])

# Função para substituir ao invés de remover a pontuação, pois dessa forma o texto mantém sua estrutura
replacePunctuation <- function(x) { gsub("[[:punct:]]+", " ", x) }
replacePunctuation("hello...world")


# Aplicação do Word stemming para remover os tempos verbais das palavras
dados_corpus_clean <- tm_map(dados_corpus_clean, stemDocument)

# Verificando a versão final do Corpus
lapply(dados_corpus[1:3], as.character)
lapply(dados_corpus_clean[1:3], as.character)


####---------------------------------------------Construção da solução---------------------------------------------------------------------####

# Criação de uma matriz esparsa para melhorar o processamento dos dados

# É possível utilizar diferentes formas para construir a matriz esparsa. Por conta disso abaixo estão duas formas.

# Essa forma utiliza os dados pré-processados
dados_dtm <- DocumentTermMatrix(dados_corpus_clean)
dados_dtm

# Essa forma aplica as transformações como parâmetros
dados_dtm2 <- DocumentTermMatrix(dados_corpus, control = list(tolower = TRUE,
                                                              removeNumbers = TRUE,
                                                              stopwords = function(x) {removeWords(x, stopwords())},
                                                              removePunctuation = TRUE,
                                                              stemming = TRUE))
dados_dtm2


# Separando o dataset em treino e teste

dados_dtm_treino <- dados_dtm[1:4169,]
dados_dtm_teste <- dados_dtm[4170:5559,]


# Variável target para cada tipo de dado
dados_train_target <- dados[1:4169, ]$type
dados_teste_target <- dados[4170:5559, ]$type


# Verificando se a proporção de Spam é similar das bases
prop.table(table(dados_train_target))
prop.table(table(dados_teste_target))


# Word Cloud para verificar a frequência das palavras
wordcloud(dados_corpus_clean, min.freq = 50, random.order = FALSE)

# As palavras call, get, can, will e come são as que mais aparecem


# Verificar a frequência dos dados
sms_dtm_freq_train <- removeSparseTerms(dados_dtm_treino, 0.999)
sms_dtm_freq_train


# Verificar quais as palavras que mais aparecem 
findFreqTerms(dados_dtm_treino, 5)


# Armazenando o resultado da função acima para criar outra base de dados
sms_freq_words <- findFreqTerms(dados_dtm_treino, 5)
str(sms_freq_words)


# Criando subsets só com palavras mais frequentes
sms_dtm_freq_train <- dados_dtm_treino[,sms_freq_words]
sms_dtm_freq_test <- dados_dtm_teste[, sms_freq_words]


# Convertendo para fator
convert_counts <- function(x) {
  x <- ifelse(x>0, 'Yes', 'No')
}


# Aplicando o apply para converter counts para colunas de dados de treino e teste
sms_train <- apply(sms_dtm_freq_train, MARGIN=2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN=2, convert_counts)


# Treinando o modelo usando laplace para resolver o problema da frequência zero
nb_classifier <- naiveBayes(sms_train, dados_train_target, laplace=1)


# Avaliando o modelo
sms_test_pred <- predict(nb_classifier, sms_test)


# Construindo a Matrix de Confusão
CrossTable(sms_test_pred,
           dados_teste_target,
           prop.chisq = FALSE,
           prop.t = FALSE,
           prop.r = FALSE,
           dnn = c('Previsto', 'Observado'))


# Com base na Matrix de Confusão o modelo obteve um ótimo resultado.Do total de 1390 observações ele classificou corretamente 1342.




