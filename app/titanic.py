# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 23:18:08 2023

@author: Alexandre
"""

import pandas as pd
import streamlit as st
import numpy as np
import time 

#                         ****** Carregamento de Dados ******

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

def clean_colum(df):
    df.drop(columns=['Unnamed: 0'], inplace=True)
    return df

# Tratamento no Dataframe
df_teino_amostra = clean_colum(df_teino_amostra)
df_teste_amostra = clean_colum(df_teste_amostra)
df_resumo_tipo_dados_treino = clean_colum(df_resumo_tipo_dados_treino)
df_dados_estatistico_unificado_treino = clean_colum(df_dados_estatistico_unificado_treino)
df_dados_passageiros_viajando_familiares_treino = clean_colum(df_dados_passageiros_viajando_familiares_treino)
df_dados_sobrevievente_pais_filhos = clean_colum(df_dados_sobrevievente_pais_filhos)
df_dados_sobrevievente_irmaos_conjugue  = clean_colum(df_dados_sobrevievente_irmaos_conjugue)

st.set_page_config(
    page_title="Dashboard Titanic",
    page_icon=":chart_with_upwards_trend:",
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

# Aplica o tema escuro usando st.markdown
st.markdown(dark_theme, unsafe_allow_html=True)
st.title("DASHBOARD TITANIC")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Visão Geral dos Dados", 
     "Modelo 1", 
     "Modelo 2", 
     "Modelo 3", 
     'Modelo 4', 
     'Modelo 5']
    )

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
           st.image("https://assets.stickpng.com/images/580b585b2edbce24c47b2472.png", width=300)

        with col2:
           st.text('Resumo Estatístico')
           st.dataframe(df_resumo_estatistico_dados_treino)
           st.text('Dados Estatístico Age e Fare')
           st.dataframe(df_dados_estatistico_unificado_treino)
           st.header('Análise de Embarque')
           st.text('Embarque por Gêneros')
           st.dataframe(df_dados_generos_treino)
           st.text('Embarque de Passageiros Viajando Sozinhos')
           st.dataframe(df_dados_passageiros_viajando_sozinho_treino)
           st.caption('1 - SIM  |  0 - NÃO')
           st.text("Embarque Geral - Família")
           st.dataframe(df_dados_passageiros_viajando_familiares_treino)
        with col3:
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
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSfy16wtIghIkk4AhaC49fdrkFIp-FYq150ew&usqp=CAU", width=300)
    with tabb:
        st.header("DADOS DE TESTE")  
    
         

   
  
       

    # with col2:
    #     st.header("DADOS DE TESTE")
    #     st.text('Amostra')
    #     st.dataframe(df_teste_amostra)
    #     st.text('Tipo de Dados')
    #     st.dataframe(df_tipo_dados_teste)

with tab2:
   st.header("Modelo 1")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("Modelo 2")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)


