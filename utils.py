from config import DATA_DIR  # Importa a configuração da pasta de dados

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
import random
import os

def calcular_mms(df, periodo):
    """
    Calcula a Média Móvel Simples (MMS).
    
    Args:
        precos (list ou pd.Series): Série de preços.
        periodo (int): Período da média móvel.
    
    Returns:
        pd.Series: Média Móvel Simples.
    """
    df[f'mms_{periodo}'] = df['close'].rolling(window=periodo).mean()
    
    return df

def calcular_mme(df, periodo):
    """
    Calcula a Média Móvel Exponencial (MME).
    
    Args:
        precos (list ou pd.Series): Série de preços.
        periodo (int): Período da média móvel.
    
    Returns:
        pd.Series: Média Móvel Exponencial.
    """
    df[f'mme_{periodo}'] = df['close'].ewm(span=periodo, adjust=False).mean()
    
    return df

def calcular_ifr(df, periodo=14):
    """
    Calcula o Índice de Força Relativa (IFR ou RSI - Relative Strength Index).
    
    Args:
        precos (list ou pd.Series): Série de preços.
        periodo (int): Período do IFR.
    
    Returns:
        pd.Series: Valores do IFR.
    """
    precos = df['close']
    variacao = precos.diff(1)
    
    ganho = variacao.where(variacao > 0, 0)
    perda = -variacao.where(variacao < 0, 0)
    
    media_ganho = ganho.rolling(window=periodo).mean()
    media_perda = perda.rolling(window=periodo).mean()
    
    rs = media_ganho / media_perda
    ifr = 100 - (100 / (1 + rs))
    df[f'ifr_{periodo}'] = ifr
    
    return df

def get_stock_data(stock):
    """
    Função para carregar os dados de uma ação usando caminho configurável.
    
    Parâmetros:
        stock (str): Nome da ação (ex: 'PETR4')
    
    Retorna:
        tuple: DataFrame com os dados e nome da coluna target
    """
    # Carrega dados do arquivo correto
    df = pd.read_csv(
        f'{DATA_DIR}{stock}_textuais_numericos.csv',  # Usa DATA_DIR do config
        encoding='utf-8-sig',
        index_col='date'
    )
    
    # Cálculo de indicadores (mantive seu código original)
    df = calcular_mms(df, 30)
    df = calcular_mms(df, 5)
    df = calcular_mme(df, 5)
    df = calcular_mme(df, 30)
    df = calcular_ifr(df)
    
    # Limpeza final
    df.dropna(inplace=True)
    
    return df

def load_stocks_data(stocks):
    """
    Carrega os dados das empresas usando a função get_stock_data e armazena-os em um dicionário.

    Parâmetros:
        stocks (list): Lista de códigos das ações.

    Retorna:
        dict: Dicionário com os códigos das ações como chave e os dados como valor.
    """
    stock_data_dict = {}
    
    for stock in stocks:
        try:
            print(f"Carregando dados para: {stock}")
            stock_data = get_stock_data(stock)  # Obtém os dados da função
            stock_data_dict[stock] = stock_data
        except Exception as e:
            print(f"Erro ao carregar dados para {stock}: {e}")
    
    return stock_data_dict

def make_combination_data(df, num_combination):
    """
    Combina dados com base em um identificador e retorna o DataFrame combinado e uma descrição.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados originais.
        num_combination (int): Número identificador da combinação.

    Retorna:
        tuple: DataFrame combinado e descrição da combinação.
    """
    # Definindo as combinações
    # Stock Data
    stock_data = ['open','close','high','low','volume']
    # Google News
    google_news = ['compound_gn', 'negative_gn', 'neutral_gn', 'positive_gn']
    # Twitter
    twitter = ['compound_tw', 'negative_tw', 'neutral_tw', 'positive_tw']
    # MMS
    mms = ['mms_30', 'mms_5']
    # MME
    mme = ['mme_5', 'mme_30']
    # IFR de 14 periodos
    ifr = ['ifr_14']
    # Valor de fechamento
    close = ['close']
    
    # Mapear combinações para descrição e colunas
    combinations = {
        1: ("Stock Data", stock_data),
        2: ("Stock Data + Google News", stock_data + google_news),
        3: ("Stock Data + Twitter", stock_data + twitter),
        4: ("Stock Data + IT (IFR + MMS + MME)", stock_data + ifr + mms + mme),
        5: ("Google News + Twitter + IFR + MMS", close + google_news + twitter + ifr + mms),
        6: ("Google News + Twitter + IFR + MME", close + google_news + twitter + ifr + mme),
        7: ("Google News + Twitter + IFR + MME + MMS", close + google_news + twitter + ifr + mme + mms),
    }

    # Verifica se a combinação existe
    if num_combination not in combinations:
        raise ValueError(f"Combinação {num_combination} não é válida. Escolha entre 1 e {len(combinations)}.")

    # Recupera descrição e colunas da combinação
    description_combination, selected_columns = combinations[num_combination]

    # Retorna o DataFrame filtrado e a descrição
    return df[selected_columns], description_combination

def define_target(df, target_column='close'):
    """
    Define a coluna de alvo para um modelo de regressão com base no preço de fechamento do próximo dia.
    
    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados.
        target_column (str): Nome da coluna que será usada como base para o cálculo do alvo. Default é 'close'.
    
    Retorna:
        pd.DataFrame: DataFrame atualizado com a nova coluna de alvo ('close_next_day') e sem a última linha.
    """
    # Trabalhar com uma cópia explícita do DataFrame para evitar avisos
    df = df.copy()
    
    # Criar a coluna de preço de fechamento do próximo dia
    df['close_next_day'] = df[target_column].shift(-1)
    
    # Remover a última linha, pois não há valor para a previsão do próximo dia
    df = df[:-1]
 
    return df, 'close_next_day'

def carregar_todos_resultados(pasta):
    """
    Carrega todos os arquivos CSV de uma pasta especificada e imprime os nomes dos arquivos carregados.

    Parâmetros:
        pasta (str): Caminho para a pasta onde os arquivos CSV estão localizados.

    Retorna:
        dict: Dicionário onde cada chave é o nome do arquivo (sem extensão) e o valor é um DataFrame com os resultados.
    """
    resultados = {}

    # Verifica se a pasta existe
    if not os.path.exists(pasta):
        print(f"Pasta '{pasta}' não encontrada.")
        return resultados

    # Itera sobre todos os arquivos na pasta
    arquivos_carregados = []
    for filename in os.listdir(pasta):
        if filename.endswith(".csv"):  # Filtra apenas arquivos .csv
            caminho_arquivo = os.path.join(pasta, filename)
            try:
                # Carrega o arquivo CSV
                df = pd.read_csv(
                    caminho_arquivo,
                    encoding='utf-8-sig',
                    sep=';'
                )
                # Nome do arquivo sem extensão
                nome_arquivo = os.path.splitext(filename)[0]
                resultados[nome_arquivo] = df
                arquivos_carregados.append(nome_arquivo)
                print(f"Carregado: {caminho_arquivo}")
            except Exception as e:
                print(f"Erro ao carregar {filename}: {str(e)}")

    # Imprime os arquivos carregados
    if arquivos_carregados:
        print("\nModelos carregados com sucesso:")
        for arquivo in arquivos_carregados:
            print(f"- {arquivo}")
    else:
        print("\nNenhum arquivo carregado.")

    return resultados

def plot_rmse_facet_plotly_normal(df):
    """
    Plota um gráfico de linhas interativo RMSE vs Stock para todas as combinações de variáveis

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as métricas de desempenho.
    """

    fig = px.line(
        df,
        x="Stock",
        y="RMSE",
        color="Description",
        facet_col="Model",
        title="RMSE X Stock",
        labels={"Stock": "Stock", "RMSE": "RMSE", "Description": "Descrição"},
        height=500, # Ajusta a altura do gráfico
        width=1100 # Ajusta a largura do gráfico
    )

    fig.update_traces(mode="lines+markers") # Adiciona linhas e marcadores

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20), # Ajusta as margens do gráfico
        font=dict(size=10), # Ajusta o tamanho da fonte
        title_x=0.5 # Centraliza o título
    )

    return fig

def plot_rmse_facet_plotly(df):
    """
    Plota um gráfico de linhas interativo RMSE vs Stock para todas as combinações de variáveis

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as métricas de desempenho.
    """

    fig = px.line(
        df,
        x="Stock",
        y="RMSE",
        color="Description",
        facet_col="Model",
        #title="RMSE X Stock",
        labels={"Stock": "Stock", "RMSE": "RMSE", "Description": "Descrição"},
        height=600,  # Ajusta a altura do gráfico
        width=1400  # Ajusta a largura do gráfico
    )

    fig.update_traces(mode="lines+markers")  # Adiciona linhas e marcadores

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=60),  # Ajusta as margens do gráfico
        font=dict(size=10),  # Ajusta o tamanho da fonte geral
        title_x=0.5,  # Centraliza o título
        legend=dict(
            yanchor="top",  # Posição vertical da legenda
            y=-0.2,  # Ajusta o deslocamento vertical da legenda
            xanchor="center",  # Posição horizontal da legenda
            x=0.5,  # Deslocamento horizontal da legenda
            orientation="h",  # Orientação horizontal da legenda
            font=dict(size=15)  # Aumenta o tamanho da fonte da legenda
        )
    )

    # Aplica as alterações aos eixos de cada gráfico gerado pelo facet_col
    for axis in fig.layout:
        if axis.startswith('xaxis') or axis.startswith('yaxis'):
            fig.layout[axis].title.font.size = 14
            fig.layout[axis].tickfont.size = 14

    # Ajusta os títulos dos facet_col
    for i, model in enumerate(df['Model'].unique()):
        fig.layout.annotations[i].font.size = 16

    return fig

def plot_rmse_line_plotly(df, stock_name, model_name=None):
    """
    Plota um gráfico de linhas interativo mostrando o RMSE para cada Description,
    filtrando por ação e, opcionalmente, por modelo selecionado.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as métricas de desempenho.
        stock_name (str): Nome da ação para filtrar os dados.
        model_name (str, opcional): Nome do modelo para filtrar os dados.
    """

    # Filtrar os dados com base na ação fornecida
    filtered_df = df[df["Stock"] == stock_name]

    # Se um modelo específico não for fornecido, plotar para todos os modelos
    if model_name is None:
        fig = px.line(
            filtered_df,
            x="Description",
            y="RMSE",
            color="Model",  # Adiciona uma cor para cada modelo
            title=f"RMSE por Combinações para {stock_name}",
            labels={"Description": "Description", "RMSE": "RMSE"},
            height=500,
            width=800,
            markers=True
        )
    else:
        # Filtrar os dados com base no modelo fornecido
        filtered_df = filtered_df[filtered_df["Model"] == model_name]

        fig = px.line(
            filtered_df,
            x="Description",
            y="RMSE",
            title=f"RMSE por Combinações para {stock_name}",
            labels={"Description": "Description", "RMSE": "RMSE"},
            height=500,
            width=800,
            markers=True
        )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=10),
        title_x=0.5
    )

    return fig

def process_stocks_and_save_metrics(all_stock_data, num_combination, name_model):
    """
    Processa uma lista de ações, treina o modelo para cada uma e salva as métricas em um DataFrame.

    Parâmetros:
        all_stock_data (dict): Dicionário com ações como chaves e DataFrames como valores.
        num_combination (int): Número da combinação de features.
        name_model (function): Função/modelo a ser treinado.

    Retorna:
        pd.DataFrame: DataFrame contendo o nome da ação, descrição dos dados, nome do modelo e as métricas.
    """
    if not isinstance(all_stock_data, dict):
        raise TypeError(f"Esperado um dicionário em 'all_stock_data', mas recebeu {type(all_stock_data).__name__}")

    stock_list = list(all_stock_data.keys())
    results = []

    for stock in stock_list:
        print(f"{name_model.__name__}: Processando {stock} | Combinação {num_combination}...")
        
        try:
            # Obtém os dados da ação
            df = all_stock_data[stock]
            
            # Realizar as combinações de features
            df, description_combination = make_combination_data(df, num_combination)
            
            # Definir target
            df, target_column = define_target(df)
            
            # Treinar modelo e coletar métricas
            metrics = name_model(stock, df, target_column, num_combination)
            
            # Salvar resultados
            results.append({
                "Stock": stock,
                "Description": description_combination,
                "Model": name_model.__name__,
                **metrics
            })
        except Exception as e:
            print(f"Erro ao processar {stock}: {str(e)}")
    
    return pd.DataFrame(results)





