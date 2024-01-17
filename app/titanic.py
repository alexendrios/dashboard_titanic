# -*- coding: utf-8 -*-
"""
@author: Alexandre

Create 17/01/2024
"""

import pandas as pd
import streamlit as st
import numpy as np
import time 


def clean_colum(df):
    df.drop(columns=['Unnamed: 0'], inplace=True)
    return df

def report_treinamento(data_frame):
    acuracia =f'''
    #####  Acurácia: {data_frame.Acuracia.values[0].round(2)}
    '''
    st.markdown(acuracia)
    st.markdown("#####  Relatório de Classificação:")   
    for i in data_frame.Relatorio_Classificacao: 
        st.code(i)

    matriz_confusao_str = data_frame.Matriz_Confusao.values[0]
    st.markdown("#####  Matriz de Confusão")
    st.code(matriz_confusao_str) 
    
def grafico_melhores_algoritmos_mlp(model1, model2, model3, lista_identificacao):
    data = [model1.Acuracia.values[0], model2.Acuracia.values[0], model3.Acuracia.values[0]]
    colunas = lista_identificacao
    df_acuracias = pd.DataFrame(data, columns=['Acurácia'], index=colunas).round(2)
    df_acuracias = df_acuracias.sort_values(by='Acurácia')
    return df_acuracias

#'''  
#     Setup:
#     1 - Carregamento dos Arquivos
#     2 - Criação dos DataFrames
#     3 - Impeza dos Dataframes    
#'''

#                         ****** Carregamento de Arquivos ******

# Treino
amostra_treino = '../out/analises/informacoes_gerais/treino/tamanho_dados_treinamento.csv'
tipo_dados_treino = '../out/analises/informacoes_gerais/treino/tipos_dados_treinamento.csv'
resumo_tipo_dados_treino = '../out/analises/informacoes_gerais/treino/tipos_dados_treinamento_resumido.csv'
dados_nulos_treino = '../out/analises/informacoes_gerais/treino/relatorio_dados_treinamento_nulos.csv'
resumo_estatistico_dados_treino = '../out/analises/informacoes_gerais/treino/resumo_estatistico_dados_treino.csv'
dados_estatistico_unificado_treino = '../out/analises/informacoes_gerais/treino/dados_estatistico_unificados_treino.csv'
dados_generos_treino = '../out/analises/informacoes_gerais/treino/analise_genero.csv'
dados_passageiros_viajando_sozinho_treino = '../out/analises/informacoes_gerais/treino/passageiros_viajando_sozinho.csv'
dados_passageiros_viajando_familiares_treino = '../out/analises/informacoes_gerais/treino/embarque_familiares.csv'
dados_sobreviventes = '../out/analises/informacoes_gerais/treino/analise_sobrevivente.csv'
dados_sobrevivente_genero = '../out/analises/informacoes_gerais/treino/analise_genero_sobrevivente.csv'
dados_sobrevivente_classe = '../out/analises/informacoes_gerais/treino/analise_classe_sobrevivente.csv'
dados_sobrevivente_classe_genero = '../out/analises/informacoes_gerais/treino/analise_classe_genero_sobrevivente.csv'
dados_sobrevievente_pais_filhos = '../out/analises/informacoes_gerais/treino/pais_filhos_taxa_sobrevivente.csv'
dados_sobrevievente_irmaos_conjugue = '../out/analises/informacoes_gerais/treino/irmaos_conjugues_taxa_sobrevivente.csv'

# Teste
amostra_teste = '../out/analises/informacoes_gerais/teste/tamanho_dados_teste.csv'
tipo_dados_teste = '../out/analises/informacoes_gerais/teste/tipos_dados_teste.csv'
resumo_tipo_dados_teste = '../out/analises/informacoes_gerais/teste/tipos_dados_teste_resumido.csv'
dados_nulos_teste = '../out/analises/informacoes_gerais/teste/relatorio_dados_teste_nulos.csv'
resumo_estatistico_dados_teste = '../out/analises/informacoes_gerais/teste/resumo_estatistico_dados_teste.csv'
dados_estatistico_unificado_teste = '../out/analises/informacoes_gerais/teste/dados_estatistico_unificados_teste.csv'
dados_generos_teste = '../out/analises/informacoes_gerais/teste/analise_genero.csv'
dados_passageiros_viajando_sozinho_teste = '../out/analises/informacoes_gerais/teste/passageiros_viajando_sozinho.csv'
dados_passageiros_viajando_familiares_teste = '../out/analises/informacoes_gerais/teste/embarque_familiares.csv'
dados_classe_genero = '../out/analises/informacoes_gerais/teste/analise_classe_genero.csv'

# modelos
modelo1_arvore_classificao = '../out/analises/modelo1/modelo1_arvore_classificacao_acuracia.csv'
modelo1_KNeighbors_classificacao = '../out/analises/modelo1/modelo1_knn_classificacao_acuracia.csv'
modelo1_regressao_logistica =  '../out/analises/modelo1/modelo1_regressao_logistica_acuracia.csv'
modelo2_arvore_classificao = '../out/analises/modelo2/modelo2_arvore_classificacao_acuracia.csv'
modelo2_KNeighbors_classificacao = '../out/analises/modelo2/modelo2_knn_classificacao_acuracia.csv'
modelo2_regressao_logistica =  '../out/analises/modelo2/modelo2_regressao_logistica_acuracia.csv'
modelo3_arvore_classificao = '../out/analises/modelo3/modelo3_arvore_classificacao_acuracia.csv'
modelo3_KNeighbors_classificacao = '../out/analises/modelo3/modelo3_knn_classificacao_acuracia.csv'
modelo3_regressao_logistica =  '../out/analises/modelo3/modelo3_regressao_logistica_acuracia.csv'
modelo4_regressao_logistica =  '../out/analises/modelo4/modelo4_regressao_logistica_acuracia.csv'
modelo4_random_florest = '../out/analises/modelo4/modelo4_random_forest_acuracia.csv'
modelo4_mlp_classificacao = '../out/analises/modelo4/modelo4_mlp_classificacao_acuracia.csv'
modelo5_regressao_logistica =  '../out/analises/modelo5/modelo5_regressao_logistica_acuracia.csv'
modelo5_random_florest = '../out/analises/modelo5/modelo5_random_forest_acuracia.csv'
modelo5_mlp_classificacao = '../out/analises/modelo5/modelo5_mlp_classificacao_acuracia.csv'

# predição
predicao_modelo1 = '../out/predicao/modelo1_sobreviventes.csv'
predicao_modelo2 = '../out/predicao/modelo2_sobreviventes.csv'
predicao_modelo3 = '../out/predicao/modelo3_sobreviventes.csv'
predicao_modelo4 = '../out/predicao/modelo4_sobreviventes.csv'
predicao_modelo5 = '../out/predicao/modelo5_sobreviventes.csv'

#                         ****** Criação dos DataFrames ******
# Dataframe Treino
df_teino_amostra = pd.read_csv(amostra_treino, sep=',')
df_tipo_dados_treino = pd.read_csv(tipo_dados_treino, sep=',')
df_resumo_tipo_dados_treino = pd.read_csv(resumo_tipo_dados_treino, sep=',')
df_dados_nulos_treino = pd.read_csv(dados_nulos_treino, sep=',')
df_resumo_estatistico_dados_treino = pd.read_csv(resumo_estatistico_dados_treino, sep=',')
df_dados_estatistico_unificado_treino = pd.read_csv(dados_estatistico_unificado_treino, sep=',')
df_dados_generos_treino = pd.read_csv(dados_generos_treino, sep=',')
df_dados_passageiros_viajando_sozinho_treino = pd.read_csv(dados_passageiros_viajando_sozinho_treino, sep=',')
df_dados_passageiros_viajando_familiares_treino = pd.read_csv(dados_passageiros_viajando_familiares_treino, sep=',')
df_dados_sobreviventes = pd.read_csv(dados_sobreviventes, sep=',')
df_dados_sobrevivente_genero = pd.read_csv(dados_sobrevivente_genero, sep=',')
df_dados_sobrevivente_classe = pd.read_csv(dados_sobrevivente_classe, sep=',')
df_dados_sobrevivente_classe_genero = pd.read_csv(dados_sobrevivente_classe_genero, sep=',')
df_dados_sobrevievente_pais_filhos = pd.read_csv(dados_sobrevievente_pais_filhos,sep=',')
df_dados_sobrevievente_irmaos_conjugue =pd.read_csv(dados_sobrevievente_irmaos_conjugue, sep=',')

# Dataframe Teste
df_teste_amostra = pd.read_csv(amostra_teste, sep=',')
df_tipo_dados_teste = pd.read_csv(tipo_dados_teste, sep=',')
df_resumo_tipo_dados_teste = pd.read_csv(resumo_tipo_dados_teste, sep=',')
df_dados_nulos_teste = pd.read_csv(dados_nulos_teste, sep=',')
df_resumo_estatistico_dados_teste = pd.read_csv(resumo_estatistico_dados_teste, sep=',')
df_dados_estatistico_unificado_teste = pd.read_csv(dados_estatistico_unificado_teste, sep=',')
df_dados_generos_teste = pd.read_csv(dados_generos_teste, sep=',')
df_dados_passageiros_viajando_familiares_teste = pd.read_csv(dados_passageiros_viajando_familiares_teste, sep=',')
df_dados_classe_genero = pd.read_csv(dados_classe_genero, sep=',')

# Dataframe modelos
df_modelo1_arvore_classificao = pd.read_csv(modelo1_arvore_classificao, sep=',')
df_modelo1_KNeighbors_classificacao = pd.read_csv(modelo1_KNeighbors_classificacao, sep=',')
df_modelo1_regressao_logistica = pd.read_csv(modelo1_regressao_logistica, sep=',')
df_modelo2_arvore_classificao = pd.read_csv(modelo2_arvore_classificao, sep=',')
df_modelo2_KNeighbors_classificacao = pd.read_csv(modelo2_KNeighbors_classificacao, sep=',')
df_modelo2_regressao_logistica = pd.read_csv(modelo2_regressao_logistica, sep=',')
df_modelo3_arvore_classificao = pd.read_csv(modelo3_arvore_classificao, sep=',')
df_modelo3_KNeighbors_classificacao = pd.read_csv(modelo3_KNeighbors_classificacao, sep=',')
df_modelo3_regressao_logistica = pd.read_csv(modelo3_regressao_logistica, sep=',')
df_modelo4_regressao_logistica = pd.read_csv(modelo4_regressao_logistica, sep=',')
df_modelo4_random_florest = pd.read_csv(modelo4_random_florest, sep=',')
df_modelo4_mlp_classificacao = pd.read_csv(modelo4_mlp_classificacao, sep=',')
df_modelo5_regressao_logistica = pd.read_csv(modelo5_regressao_logistica, sep=',')
df_modelo5_random_florest = pd.read_csv(modelo5_random_florest, sep=',')
df_modelo5_mlp_classificacao = pd.read_csv(modelo5_mlp_classificacao, sep=',')

#predicao
df_predicao_modelo1 = pd.read_csv(predicao_modelo1, sep=',')
df_predicao_modelo2 = pd.read_csv(predicao_modelo2, sep=',')
df_predicao_modelo3 = pd.read_csv(predicao_modelo3, sep=',')
df_predicao_modelo4 = pd.read_csv(predicao_modelo4, sep=',')
df_predicao_modelo5 = pd.read_csv(predicao_modelo5, sep=',')

# Tratamento no Dataframe
# treino
df_teino_amostra = clean_colum(df_teino_amostra)
df_resumo_tipo_dados_treino = clean_colum(df_resumo_tipo_dados_treino)
df_dados_estatistico_unificado_treino = clean_colum(df_dados_estatistico_unificado_treino)
df_dados_passageiros_viajando_familiares_treino = clean_colum(df_dados_passageiros_viajando_familiares_treino)
df_dados_sobrevievente_pais_filhos = clean_colum(df_dados_sobrevievente_pais_filhos)
df_dados_sobrevievente_irmaos_conjugue  = clean_colum(df_dados_sobrevievente_irmaos_conjugue)
df_dados_passageiros_viajando_sozinho_teste = pd.read_csv(dados_passageiros_viajando_sozinho_teste, sep=',')

# Teste
df_teste_amostra = clean_colum(df_teste_amostra)
df_resumo_tipo_dados_teste = clean_colum(df_resumo_tipo_dados_teste) 
df_dados_estatistico_unificado_teste = clean_colum(df_dados_estatistico_unificado_teste)
df_dados_passageiros_viajando_familiares_teste = clean_colum(df_dados_passageiros_viajando_familiares_teste)

# modelos
df_modelo1_arvore_classificao = clean_colum(df_modelo1_arvore_classificao) 
df_modelo1_KNeighbors_classificacao  = clean_colum(df_modelo1_KNeighbors_classificacao)
df_modelo1_regressao_logistica = clean_colum(df_modelo1_regressao_logistica)
df_modelo2_arvore_classificao = clean_colum(df_modelo2_arvore_classificao) 
df_modelo2_KNeighbors_classificacao  = clean_colum(df_modelo2_KNeighbors_classificacao)
df_modelo2_regressao_logistica = clean_colum(df_modelo2_regressao_logistica)
df_modelo3_arvore_classificao = clean_colum(df_modelo3_arvore_classificao) 
df_modelo3_KNeighbors_classificacao  = clean_colum(df_modelo3_KNeighbors_classificacao)
df_modelo3_regressao_logistica = clean_colum(df_modelo3_regressao_logistica)
df_modelo4_regressao_logistica = clean_colum(df_modelo4_regressao_logistica)
df_modelo4_random_florest = clean_colum(df_modelo4_random_florest)
df_modelo4_mlp_classificacao = clean_colum(df_modelo4_mlp_classificacao)
df_modelo5_regressao_logistica = clean_colum(df_modelo5_regressao_logistica)
df_modelo5_random_florest = clean_colum(df_modelo5_random_florest)
df_modelo5_mlp_classificacao = clean_colum(df_modelo5_mlp_classificacao)

# predição
df_predicao_modelo1 = clean_colum(df_predicao_modelo1)
df_predicao_modelo2 = clean_colum(df_predicao_modelo2)
df_predicao_modelo3 = clean_colum(df_predicao_modelo3)
df_predicao_modelo4 = clean_colum(df_predicao_modelo4)
df_predicao_modelo5 = clean_colum(df_predicao_modelo5)

#'''  
#    Desenvolvimento:
#    1 - Configuração 
#    2 - Criação do Dashboard
#'''

#Configuração da Página do Dasboard
ico ="../ico/titanic.ico" 

st.set_page_config (
    page_title="Dashboard Titanic",
    page_icon=ico,
    layout="wide",
    initial_sidebar_state="expanded"
)

dark_theme = """
    <style>
        body {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .css-1v3fvcr {
            color: #FFFFFF;
        }
    </style>
"""
st.markdown(dark_theme, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center'>📊DASHBOARD TITANIC</h2>", unsafe_allow_html=True)
st.markdown('---')
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Visão Geral dos Dados", 
     "Modelo 1", 
     "Modelo 2", 
     "Modelo 3", 
     'Modelo 4', 
     'Modelo 5',
     'Predição']
    )

# Apresentação das Análises
with tab1:
    taba, tabb = st.tabs(["Dados de Treino", "Dados de Teste"])
    with taba:
        st.header("DADOS DE TREINO - VISÃO GERAL")
        col1, col2, col3 = st.columns(3)
        with col1:
           st.text('Amostra')
           st.dataframe(df_teino_amostra)
           st.text('Tipo de Dados')
           st.dataframe(df_tipo_dados_treino)
           st.text('Resumo Tipo de Dados')
           st.dataframe(df_resumo_tipo_dados_treino)
           st.text('Visão dos Dados Nulos')
           st.dataframe(df_dados_nulos_teste)
           st.image("../images/dados.png", width=300)

        with col2:
           st.text('Resumo Estatístico')
           st.dataframe(df_resumo_estatistico_dados_treino)
           st.text('Dados Estatístico Age e Fare')
           st.dataframe(df_dados_estatistico_unificado_treino)
           st.header('Análise de Embarque')
           st.text('Embarque por Gênero')
           st.dataframe(df_dados_generos_treino)
           st.text('Embarque de Passageiros Viajando Sozinhos')
           st.dataframe(df_dados_passageiros_viajando_sozinho_treino)
           st.caption('1 - SIM  |  0 - NÃO')
           st.text("Embarque Geral - Família")
           st.dataframe(df_dados_passageiros_viajando_familiares_treino)
        with col3:
            st.image("../images/titanic.jpeg", 
                     width=300)
            st.header('Análise Sobreviventes')
            st.caption('Sobreviventes: 1 - SIM  |  0 - NÃO')
            st.text('Dados Sobreviventes')
            st.dataframe(df_dados_sobreviventes)
            st.text('Sobrevivente por Gênero')
            st.dataframe(df_dados_sobrevivente_genero)
            st.text('Sobrevivente por Classe')
            st.dataframe(df_dados_sobrevivente_classe)
            st.text('Sobrevivente por Classe e Gênero')
            st.dataframe(df_dados_sobrevivente_classe_genero)
            st.text('Sobrevivente Pais/Filhos')
            st.dataframe(df_dados_sobrevievente_pais_filhos)
            st.text('Sobrevivente Irmãos/Conjugue')
            st.dataframe(df_dados_sobrevievente_irmaos_conjugue )
            
    with tabb:
        st.header("DADOS DE TESTE - VISÃO GERAL")  
        col1, col2,col3 = st.columns(3)
        with col1:
            st.text('Amostra')
            st.dataframe(df_teste_amostra)
            st.text('Tipo de Dados')
            st.dataframe(df_tipo_dados_teste)
            st.text('Resumo Tipo de Dados')
            st.dataframe(df_resumo_tipo_dados_teste)
            st.text('Visão dos Dados Nulos')
            st.dataframe(df_dados_nulos_teste)
            st.image("../images/dados_teste.png", width=300)
        with col2:
            st.text('Resumo Estatístico')
            st.dataframe(df_resumo_estatistico_dados_teste)
            st.text('Dados Estatístico Age e Fare')
            st.dataframe(df_dados_estatistico_unificado_teste)
            st.header('Análise de Embarque')
            st.text('Embarque por Gênero')
            st.dataframe(df_dados_generos_teste)
            st.text('Embarque de Passageiros Viajando Sozinhos')
            st.dataframe(df_dados_passageiros_viajando_sozinho_teste)
            st.caption('1 - SIM  |  0 - NÃO')
            st.text("Embarque Geral - Família")
            st.dataframe(df_dados_passageiros_viajando_familiares_teste)
        with col3:
            st.image("../images/titanic_embarque.jpg", 
                     width=400)
            st.header('Detalhamento dos Dados de Embarque')
            st.text('Classe por Gênero')
            st.dataframe(df_dados_classe_genero)
            texto = '''
               Este conjunto de Dados tem o objetivo principal de:
                
               testar o modelo de Aprendizado de Máquina que obtiver
               a melhor acurácia e realizar a previsão dos sobreviventes,
               pois, neste modelo o mesmo não é apresentado
            '''
            st.header(texto)
            
with tab2: 
    data_modelo1 = '''
    ## O modelo 1 - modelo simples o mesmo foi trabalhado com três Modelos de Machine Learning:
    * Árvore de Classificação
    * KNeighborsClassifier
    * Regressão Logística

    > Estratégia utilizada Tratamentos dos Dados Nulos
    
    '''
    st .markdown(data_modelo1)
    st.text('Acurácia')
    st.bar_chart(grafico_melhores_algoritmos_mlp(
        df_modelo1_arvore_classificao,
        df_modelo1_KNeighbors_classificacao,
        df_modelo1_regressao_logistica,
        ['Árvore de Classificação', 'KNN Classificação', 'Regressão Logística']
    ),  height=300)
   
    col1, col2,col3 = st.columns(3)
    with col1:  
        st.markdown("####  Árvore de Classificação")
        report_treinamento(df_modelo1_arvore_classificao)
    with col2:
        st.markdown("####  KNeighborsClassifier")
        report_treinamento(df_modelo1_KNeighbors_classificacao)  
    with col3:
        st.markdown("####  Regressão Logística")
        report_treinamento(df_modelo1_regressao_logistica)           

with tab3:
    data_modelo2 = '''
    # O modelo 2 - foi trabalhado com três Modelos de Machine Learning:
    * Árvore de Classificação
    * KNeighborsClassifier
    * Regressão Logística

    > Tratamento das informações de texto:
    > ### Utilizando as Técnicas:
    > * A coluna “Sex”, podemos criar uma coluna chamada “Male_Check” que vai receber 1 se o gênero for masculino e 0 se o gênero for feminino
    > * A coluna Embarked usando o OneHotEncoder 
            
    '''
    st .markdown(data_modelo2)
    st.text('Acurácia')
    st.bar_chart(grafico_melhores_algoritmos_mlp(
            df_modelo2_arvore_classificao,
            df_modelo2_KNeighbors_classificacao,
            df_modelo2_regressao_logistica, 
            ['Árvore de Classificação', 'KNN Classificação', 'Regressão Logística']
        ),  height=300)
    col1, col2,col3 = st.columns(3)
    with col1:  
        st.markdown("####  Árvore de Classificação")
        report_treinamento(df_modelo2_arvore_classificao)
    with col2:
        st.markdown("####  KNeighborsClassifier")
        report_treinamento(df_modelo2_KNeighbors_classificacao)  
    with col3:
        st.markdown("####  Regressão Logística")
        report_treinamento(df_modelo2_regressao_logistica)   
     
with tab4:      
    data_modelo3 = '''
   # O modelo 3 - foi trabalhado com três Modelos de Machine Learning:
   * Árvore de Classificação
   * KNeighborsClassifier
   * Regressão Logística

   > Engenharia de Variáveis/Recursos:
   > ### Utilizando as Técnicas:
   > * Ajuste das escalas referente as coluna **'Age'** e **'Fare'**
   >  - Utilizando o RobustScaler
   > * Entendendo as colunas SibSp e Parch
   * Selecionar os melhores Recursos
   * Carregando os Dados Embarques
   * Criando o Encoder destes dados 
   * Eliminando as Colunas ['Embarked_C','Embarked_Q','Embarked_S']
        
  '''
    st .markdown(data_modelo3)
    st.text('Acurácia')
    st.bar_chart(grafico_melhores_algoritmos_mlp(
            df_modelo3_arvore_classificao,
            df_modelo3_KNeighbors_classificacao,
            df_modelo3_regressao_logistica, 
            ['Árvore de Classificação', 'KNN Classificação', 'Regressão Logística']
        ),  height=300)
    col1, col2,col3 = st.columns(3)
    with col1:  
        st.markdown("####  Árvore de Classificação")
        report_treinamento(df_modelo3_arvore_classificao)
    with col2:
        st.markdown("####  KNeighborsClassifier")
        report_treinamento(df_modelo3_KNeighbors_classificacao)  
    with col3:
        st.markdown("####  Regressão Logística")
        report_treinamento(df_modelo3_regressao_logistica)   
     
with tab5:
     data_modelo4 = '''
    ## Modelo 4:  foi trabalhado com três Modelos de Machine Learning:
    * Regressão Logística
    * Random Forest
    * MLPClassifier (Redes Neurais)

    >  #### A estratégia utilizada:
    >  * Teste Modelos e Melhora de Parâmetro
        
  '''
     st .markdown(data_modelo4)
     st.bar_chart(grafico_melhores_algoritmos_mlp(
            df_modelo4_regressao_logistica,
            df_modelo4_random_florest,
            df_modelo4_mlp_classificacao, 
            ['Regressão Logística', 'Random Florest', 'MLP Classificação']
        ),  height=300)
     col1, col2,col3 = st.columns(3)
     with col1:  
        st.markdown("####  Regressão Logística")
        report_treinamento(df_modelo4_regressao_logistica)
     with col2:
        st.markdown("####  Random Florest")
        report_treinamento(df_modelo4_random_florest)  
     with col3:
        st.markdown("####  MLP Classificação")
        report_treinamento(df_modelo4_mlp_classificacao)

with tab6:
    data_modelo5 = '''
    ## Modelo 5:  foi trabalhado com três Modelos de Machine Learning:
    * Regressão Logística
    * Random Forest
    * MLPClassifier (Redes Neurais)

    >  #### A estratégia utilizada:
    >  * Grid Search
        
  '''
    st.markdown(data_modelo5)
    st.bar_chart(grafico_melhores_algoritmos_mlp(
            df_modelo5_regressao_logistica,
            df_modelo5_random_florest,
            df_modelo5_mlp_classificacao, 
            ['Regressão Logística', 'Random Florest', 'MLP Classificação']
        ),  height=300)
    
    col1, col2,col3 = st.columns(3)
    with col1:  
        st.markdown("####  Regressão Logística")
        report_treinamento(df_modelo5_regressao_logistica)
    with col2:
        st.markdown("####  Random Florest")
        report_treinamento(df_modelo5_random_florest)  
    with col3:
        st.markdown("####  MLP Classificação")
        report_treinamento(df_modelo5_mlp_classificacao)

with tab7:
    data_modelo6 = '''
    ## Predição: 
    >  * De acordo com os algoritmos treinados nos 05 modelos
    >  * Irá ser imputado o resultado desse treinamento onde obteve a melhor acurácia
    >  * E realizar a predição com os Dados de teste, onde:
    >  * Total Vivos e Total Não vivos 
        
  '''
    st.markdown(data_modelo6)
    
    col1, col2,col3, col4, col5 = st.columns(5)
    with col1:  
        st.markdown("#####  Modelo 1 - Regressão Logística")
        st.table(df_predicao_modelo1)
    with col2:
        st.markdown("#####  Modelo 2 - Regressão Logística")
        st.table(df_predicao_modelo2)  
    with col3:
        st.markdown("#####  Modelo 3 - Regressão Logística")
        st.table(df_predicao_modelo3)
        st.image("../images/cemiterio.png", width=800)
    with col4:
        st.markdown("#####  Modelo 4 - MLP Classificação")
        st.table(df_predicao_modelo4)  
    with col5:
        st.markdown("#####  Modelo 5 - Regressão Logística")
        st.table(df_predicao_modelo5)
        

    
# Remove Estilo Streamlit
remove_st_estilo = '''  
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
'''
st.markdown(remove_st_estilo, unsafe_allow_html=True)