# Análsie de Dados e Manipulação com Numpy
# Dataset: Per Capita Income by County (2021) vs. Education
# Disponível em Kaggle
# Link: https://www.kaggle.com/datasets/ruddygunawan/per-capita-income-by-county-2021-vs-education


# Objetivo:
# Aplicar as tecnincas de manipulação de Arrays com Numpy

# Exercícios baseado nos exemplos das aulas do curso 2.Big Data Real-Time Analytics com Python e Spark
# da Data Science Academy


# Importando Biblioteca Numpy
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# Configuração de impressão do Numpy
np.set_printoptions(suppress = True, linewidth= 200, precision= 2)


# Carregando o Dataset
df = np.genfromtxt("education_vs_per_capita_personal_income.csv",
                   delimiter= ",",
                   skip_header= 1,
                   autostrip = True,
                   usecols= (0,1,3,4,5,6,7,8,9,10),
                   encoding="UTF-8")

# Visualizando os Dados
print("Imprimindo o Dataframe: ")
print(df)
print() # Solta uma linha

print("Tipo de Dado (df): ")
print(type(df))

print("Shape do DataFrame(df): ")
print(df.shape)
print() # Solta uma linha

# Verificando Valores nan
print("Valores Ausentes em df: ")
print(np.isnan(df).sum())
print() # Solta uma linha


# Tratando os Valores Ausêntes
# Criando um um valor coringo, que irá substituir os valores "nan" provisoriamente
coringa = np.nanmax(df) + 1
print("Valor coringa: ")
print(coringa)
print() # Solta uma linha


# Seprando as variáveis numéricas das variáveis do tipo string, calculando a média das variáveis númericas
media_no_nan = np.nanmean(df, axis= 0 )
print("Média das variáveis numéricas nas colunas: ")
print(media_no_nan)

print("Tipo de Dado: ")
print(type(media_no_nan))
print() # Solta uma linha


# Separando as Variáveis do Tipo String
col_str_nan = np.argwhere(np.isnan(media_no_nan)).squeeze()
print("Colunas Strings: ")
print(col_str_nan)

print("Tipo de Dado: ")
print(type(col_str_nan))
print() # Solta uma linha


# Separando as colunas Numéricas
col_num = np.argwhere(np.isnan(media_no_nan) == False).squeeze()
print("Colunas Numéricas: ")
print(col_num)

print("Tipo de Dado: ")
print(type(col_num))
print() # Solta uma linha


# Salvando as Variáveis do Tipo String em uma Variável
strings = np.genfromtxt("education_vs_per_capita_personal_income.csv",
                   delimiter= ",",
                   skip_header= 1,
                   autostrip = True,
                   usecols = col_str_nan,
                   dtype  = str,
                       encoding="UTF-8")

# Visualizando o Dataframe de Strings
print("Dataframe com os valores tipo caracter: ")
print(strings)
print() # Solta uma linha

print("Tipo de Dado: ")
print(type(strings))

print("Shape do DataFrame: ")
print(strings.shape)
print() # Solta uma linha


# Salvando as Variáveis do Tipo Numérico e preenchendo os Valores "nan"
numerics = np.genfromtxt("education_vs_per_capita_personal_income.csv",
                   delimiter= ",",
                   skip_header= 1,
                   autostrip = True,
                   usecols = col_num,
                   filling_values= coringa,
                   encoding="UTF-8")

# Visualizando o Dataframe de Strings
print("Dataframe com os valores numéricos: ")
print(numerics)
print() # Solta uma linha

print("Tipo de Dado: ")
print(type(numerics))

print("Shape do DataFrame: ")
print(numerics.shape)
print() # Solta uma linha


# Carregando os Nomes das Colunas
col_names = np.genfromtxt("education_vs_per_capita_personal_income.csv",
                   delimiter= ",",
                   autostrip = True,
                   skip_footer= df.shape[0],
                   dtype = str)


# Visualizando o Dataframe de Strings
print("Cabeçalho do conjunto de dados: ")
print(col_names)
print() # Solta uma linha

print("Tipo de Dado: ")
print(type(col_names))

print("Shape do array do cabeçalho: ")
print(col_names.shape)
print() # Solta uma linha


# Separando o Cabeçalho de Colunas Numéricas e Colunas de Strings
col_names_str = col_names[col_str_nan]

print("Nome das colunas na variável: strings ")
print(col_names_str)
print() # Solta uma linha

print("Nome das colunas na variável: numerics ")
col_names_num = np.delete(col_names, [1,2,3])
print(col_names_num)
print() # Solta uma linha


# Criando Checkpoint
def checkpoint(file_name, checkpoint_hader, checkpoint_data):
    np.savez(file_name, header = checkpoint_hader, data = checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return(checkpoint_variable)

# Chamando a Função acima
checkpoint_inicial = checkpoint("Primeiro_checkpoint", col_names_str, strings)


# Checando se os Dados são os mesmos
print("Verificando se os dados do Checkpoint criado é igual ao do array de strings acima: ")
print(np.array_equal(checkpoint_inicial["data"], strings))
print() # Solta uma linha


# Manipulando as Colunas do Tipo String
# Imprimindo a Variável "col_names_str"
print("Imprimindo a variável: col_names_str")
print(col_names_str)
print() # Solta uma linha

print("Imprimindo a variável: strings")
print(strings)
print() # Solta uma linha


# Pré-Processando a Variável "state" com Categorização
print("Imprimindo a variável com a contagem: strings")
print(np.unique(strings,return_counts=True ))
print() # Solta uma linha

# Extraindo o Nome dos Estados e suas respectivas contagens
names_states, count_state = np.unique(strings,return_counts=True)

# Ordenando em ordem decrescente
states_ordened = np.argsort(-count_state)

# Resultados Ordenados
print("Ordenação dos estados por contagem decrescente: ")
print(names_states[states_ordened], count_state[states_ordened])

# Separando os Estados por Região
west = np.array(['WA', 'OR','CA','NV','ID','MT', 'WY','UT','CO', 'AZ','NM','HI','AK'])
south = np.array(['TX','OK','AR','LA','MS','AL','TN','KY','FL','GA','SC','NC','VA','WV','MD','DE','DC'])
midwest = np.array(['ND','SD','NE','KS','MN','IA','MO','WI','IL','IN','MI','OH'])
east = np.array(['PA','NY','NJ','CT','MA','VT','NH','ME','RI'])

# Categorizando a Região.

strings[0:,] = np.where(np.isin(strings[0:,], west), 1, strings[0:,])
strings[0:,] = np.where(np.isin(strings[0:,], south), 2, strings[0:,])
strings[0:,] = np.where(np.isin(strings[0:,], midwest), 3, strings[0:,])
strings[0:,] = np.where(np.isin(strings[0:,], east), 4, strings[0:,])
print() # Solta uma linha

# Visualizando os Valores Únicos "strings"
print("Valores únicos da variável: strings ")
print(np.unique(strings[:,0]))
print() # Solta uma linha

# Separando as colunas, e salvando apenas a colua de interesse, a coluna que foi feita a categorização, referente aos
# estados separados por região geografica
strings_1 = strings[:,0]
strings = strings_1

print("Tipo de Dado (strings): ")
print(strings_1.dtype)
print() # Solta uma linha

print("Shape do DataFrame (strings): ")
print(strings_1.shape)


# O tipo da variável "strings" ainda continua como caracter, é necessário transformar para o tipo numérico
strings = strings.astype(int)
print("Tipo de Dado (strings), pós conversão: ")
print(strings.dtype)
print() # Solta uma linha


# Criando o Segundo Checkpoint, chamando a Função criando anteriormente
checkpoint_intermediario = checkpoint("Segundo_checkpoint", col_names_str, strings)


# Trabalahndo com as colunas numéricas:

# Para esta análise, vou criar uma nova coluna com a médias dos rendas per-capita dos anos 2019, 2020, 2021
print("Médias dos rendas per-capita dos anos 2019, 2020, 2021: ")
# Salvando as colunas que serçao usadas para o calculo da média em uma variável
media_per_capita = numerics[:,2:5]

# Claculando a média das colunas da variável "media_per_carpita"
media = np.mean(media_per_capita, axis=1)
print(media)
print() # Solta uma linha

# Deletando as colunas de renda per-capita
numerics = np.delete(numerics, [1,2,3,4], axis=1)

# Correção: Quando o dataframe foi semparado em colunas tipo string e numeric, a ultima coluna do dataframe original (df)
# não entrou na formação
df_9 = df[:,9]

# Adicionando coluna faltante, "percentage of bachelor's degree in the county. From 2015 to 2019."
numerics = np.column_stack((numerics, df_9))


# Adicionando a coluna "media"
print("Adicionando a coluna das médias de renda per-capita dos anos de 2019-2021")
numerics = np.insert(numerics, 1, media, axis=1)

print("Shape do DataFrame numerics: ")
print(numerics.shape)
print() # Solta uma linha

# Função para head
def head(numerics, n=8):
    return numerics[:n]

print("Head (8): ")
print(head(numerics))
print() # Solta uma linha


# Calculando a média de Graduados no país:
print("Média associate_degree período 2016 - 2020: ")
media_graduados_20 = np.mean(numerics[:,4])
print(media_graduados_20)
print() # Solta uma linha

print("Média bachelor_degree período 2015 - 2019: ")
media_graduados_19 = np.mean(numerics[:,5])
print(media_graduados_19)
print() # Solta uma linha


# Ajustando as variáveis do cabeçalho
print("Cabeçalho strings: ")
print("Antes: ")
print(col_names_str)
print() # Solta uma linha

col_names_str = ["states_by_region"]
print("Depois: ")
print(col_names_str)
print() # Solta uma linha

print("Cabeçalho numerics: ")
print("Antes: ")
print(col_names_num)
print() # Solta uma linha

print("Depois: ")
col_names_num = ["county_FIPS", "mean_pc_2019-2021", "associate_degree_numbers_2016_2020", "bachelor_degree_numbers_2016_2020",
                 "associate_degree_percentage_2016_2020", "bachelor_degree_percentage_2015_2019"]
print(col_names_num)
print() # Solta uma linha



# Criando o terceiro Checkpoint para as variáveis col_names_str e strings, chamando a Função criando anteriormente.
checkpoint_strings = checkpoint("Terceiro_checkpoint", col_names_str, strings)

# Criando o último Checkpoint para as variáveis col_names_num e numerics, chamando a Função criando anteriormente.
checkpoint_numerics = checkpoint("Ultimo_checkpoint", col_names_num, numerics)


# Criando o Dataframe Final
print("Criando o Dataframe Final")
print() # Solta uma linha

print("Dataframe com String: ")
print(checkpoint_strings['data'].shape)
print() # Solta uma linha

print("Dataframe com Numéricos: ")
print(checkpoint_numerics['data'].shape)
print() # Solta uma linha

# Dataframe Final
df_final = np.column_stack((checkpoint_numerics['data'], checkpoint_strings['data']))

# Verifica se tem valor ausente
print("Verificando se existe, valores nan")
print(np.isnan(df_final).sum())
print() # Solta uma linha

# Concatena os arrays de nomes de colunas
cabecalho_completo = np.concatenate((checkpoint_numerics['header'], checkpoint_strings['header']))
print("Cabeçalho Completo: ")
print(cabecalho_completo)

# Concatena todos os arrays em um único Dataset
df_final = np.vstack((cabecalho_completo , df_final))

print("Head (8): ")
print(head(df_final))

# Salvando o Dataset em disco
np.savetxt("Dataset_Final.csv",
           df_final,
           fmt = '%s',
           delimiter = ',')