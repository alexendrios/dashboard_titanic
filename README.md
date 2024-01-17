# dashboard_titanic

## Sobre

> Este tem a Finalidade de analisar os dados do Titanic com os seguintes objetivos:
>  - Verificar os Dados Estatpsiticos
>  - Treinar Modelos de Machine Learning
>  - Realizar a Predição dos Dados

## Estrutura do Projeto


```
C:\PROJETO_ESTUDO\PYTHON\DASHBOARD_TITANIC
├───analises
├───app
├───data
│   ├───default
│   └───modificados
├───ico
├───images
└───out
    ├───analises
    │   ├───informacoes_gerais
    │   │   ├───teste
    │   │   └───treino
    │   ├───modelo1
    │   ├───modelo2
    │   ├───modelo3
    │   ├───modelo4
    │   └───modelo5
    ├───models
    ├───predicao
    └───reports

```
> Nota - se que os dados das análies encontra - se na pasta 'analises', ao entrar nessa pasta encontrará um arquivo - titanic.ipynb - este conterá todas as aanálises, treinamento de modelos.
> Na pasta 'app' conterá o arquivo - titanic.py - com a finalidade de montar o dashboard, com todas as imformações pertinentes das análises.
> No diretório data, estará os arquivos csv, que servirá de base para os estudos.
> No diretório 'out', estarão os arquivos gerados na análise.

### Setup
> instalação do python
> instalação do anaconda
> instalação da biblioteca:
> - seaborn
> - matplotlib
> - pandas_profiling
> - sklearn
> - streamlit

### Execução
>  - local
>  - o console apntando para a raiz do projeto:
>  -  C:\PROJETO_ESTUDO\PYTHON\DASHBOARD_TITANIC
>  * digitar o seguinte comando:
```
jupyter notebook
```
> - entar no diretório analises
> - clicar no arquivo titanic.ipynb
> - Poderá abrir com o google Colab
> - VSCODE se tiver os plugins instalados
> - Entre outros

## Execução do Dashboard
> - Está no diretório do "app"
> - C:\projeto_estudo\python\dashboard_titanic\app
> - Digitar o seguinte comando:
```
streamlit run titanic.py
```
[Clique aqui para acessar o arquivo do projeto titanic.ipynb](./analises/titanic.ipynb)
