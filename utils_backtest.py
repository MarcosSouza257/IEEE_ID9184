from config import *
from utils import *

def make_predictions(stock, df, model='model_2', num_combination=4, output_dir="output"):
    """
    Carrega modelos treinados, faz previsões nos dados de teste e retorna um DataFrame 
    com os valores reais e previstos desnormalizados.

    Parâmetros:
        stock (str): Nome do ativo.
        df (pd.DataFrame): DataFrame contendo os dados para previsão.
        model (str): Nome do modelo salvo. Padrão: 'model_2'.
        num_combination (int): Identificador da combinação de features usada no treinamento.
        output_dir (str): Diretório onde os modelos e scalers estão salvos.

    Retorna:
        pd.DataFrame: DataFrame contendo os dados de teste com a coluna "close_predicted" desnormalizada.
    """

    df_results = pd.DataFrame()  # DataFrame vazio para armazenar os resultados

    try:
        # Caminhos dos arquivos salvos
        model_path = os.path.join(OUTPUT_DIR, f"{stock}\{model}_comb_{num_combination}.h5")
        scaler_X_path = os.path.join(OUTPUT_DIR, f"{stock}\{model}_comb_{num_combination}_scaler_X.pkl")
        scaler_y_path = os.path.join(OUTPUT_DIR, f"{stock}\{model}_comb_{num_combination}_scaler_y.pkl")  # Scaler para y

        # Carregando o modelo treinado e os scalers
        model = load_model(model_path)
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)  # Scaler para desnormalizar y

        # Realizar a combinação de features escolhida
        df, description_combination = make_combination_data(df, num_combination)
            
        # Definir target
        df, target_column = define_target(df)

        # Separando variáveis independentes (X) e dependente (y)
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values

        # Dividindo os dados (80% treino, 20% teste)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Normalizando os dados de teste
        X_test_scaled = scaler_X.transform(X_test)
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))  # Reshape para LSTM

        # Fazendo previsões no conjunto de teste
        y_pred_scaled = model.predict(X_test_scaled).flatten()

        # Desnormalizando as previsões
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        # Criando um DataFrame com os dados de teste e as previsões desnormalizadas
        df_results = df.iloc[len(X_train):].copy()  # Pegando apenas os 20% finais (testes)
        df_results["close_predicted"] = y_pred

    except Exception as e:
        print(f"Erro ao processar {stock}: {e}")

    return df_results

def generate_signals(df):
    """
    Gera sinais de compra e venda com base na comparação entre 'close' e 'close_predicted'.

    Parâmetros:
        stock (str): Nome da ação para referência.
        df (pd.DataFrame): DataFrame contendo as colunas 'close' e 'close_predicted'.

    Retorna:
        pd.DataFrame: DataFrame atualizado com a coluna 'signal'.
    """
    # Garantindo que a coluna 'signal' seja criada corretamente
    df["signal"] = 0  

    # Aplicando as regras para gerar sinais
    df.loc[df["close"] < df["close_predicted"], "signal"] = 1   # Sinal de compra
    df.loc[df["close"] > df["close_predicted"], "signal"] = -1  # Sinal de venda

    return df

def run_backtest(signals_df, initial_money=10000):
    """
    Executa um backtest com base nos sinais de compra, mantendo posições enquanto os sinais forem de compra.
    As operações são executadas no preço de abertura do próximo dia.
    Retorna o resultado final e um DataFrame com a curva de capital, usando o índice do DataFrame como data.

    Parâmetros:
        signals_df (pd.DataFrame): DataFrame contendo as colunas 'signal', 'close' e 'open', com um índice de data.
        initial_money (float): Quantia inicial em dinheiro para o backtest. Padrão: 10.000.

    Retorna:
        tuple: Uma tupla contendo o valor final do portfólio e um DataFrame com a curva de capital.
    """
    # Verificando se as colunas necessárias estão no DataFrame
    required_columns = {'signal', 'close', 'open'}
    if not required_columns.issubset(signals_df.columns):
        raise ValueError(f"O DataFrame precisa conter as colunas {required_columns}")

    money = initial_money
    stock = 0  # Quantidade de ações compradas
    position = 0  # 1 para comprado, 0 para neutro
    
    capital_curve = {'date': [], 'capital': []}  # Dicionário para armazenar a evolução do capital

    try:
        for i, (date, row) in enumerate(signals_df.iterrows()):  # Iterando sobre o índice e as linhas
            # Atualiza a curva de capital com o valor atual do portfólio
            if position == 1:
                capital_curve['capital'].append(stock * signals_df['close'].iloc[i]) # calcula capital com o fechamento do dia
            else:
                capital_curve['capital'].append(money)

            capital_curve['date'].append(date)  # Usando o índice como data

            if len(capital_curve['capital']) == 0:
                capital_curve['capital'].append(money)
                capital_curve['date'].append(date)

            if row['signal'] == 1 and position != 1:  # Se o sinal for de compra e não estiver comprado
                if position == 0:  # Se estiver neutro, compra
                    if i + 1 < len(signals_df): # Verifica se i+1 está dentro dos limites
                        stock = money / signals_df['open'].iloc[i+1]  # Compra no preço de abertura do dia seguinte
                        money = 0
                position = 1  # Define posição como comprada

            elif row['signal'] != 1 and position == 1: # Se o sinal não for de compra e estiver comprado, vende tudo.
                if i + 1 < len(signals_df): # Verifica se i+1 está dentro dos limites
                    money += stock * signals_df['open'].iloc[i+1]  # Vende no preço de abertura
                    stock = 0
                position = 0 # define posição como neutro

        capital_df = pd.DataFrame(capital_curve)
        capital_df.set_index('date', inplace=True) # define a coluna data como index

        return money, capital_df
    except Exception as e:
        print(f"Erro no backtest: {e}")

def plot_backtest_comparison(stock, capital_df, signals_df, initial_money=10000):
    """
    Plota a curva de capital do backtest e a evolução de um investimento inicial no ativo, usando Plotly.

    Parâmetros:
        stock (str): Nome do ativo.
        capital_df (pd.DataFrame): DataFrame com a curva de capital do backtest, com índice de data.
        signals_df (pd.DataFrame): DataFrame com os preços de fechamento, com índice de data.
        initial_money (float): Quantia inicial em dinheiro para o investimento de comparação. Padrão: 10.000.
    """

    # Criando o gráfico de linhas para a curva de capital do backtest
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=capital_df.index, y=capital_df['capital'], mode='lines', name='Strategy with Model 2'))

    # Calculando a evolução do investimento inicial no ativo
    initial_investment = [initial_money]
    for i in range(1, len(signals_df)):
        initial_investment.append(initial_money * (signals_df['close'].iloc[i] / signals_df['close'].iloc[0]))

    # Adicionando o gráfico de linhas para a evolução do investimento inicial
    fig.add_trace(go.Scatter(x=signals_df.index, y=initial_investment, mode='lines', name='Buy & Hold'))

    # Atualizando o layout do gráfico
    fig.update_layout(
        title=f'{stock} - Strategy with Model 2 vs. Buy and Hold Strategy',
        title_x=0.5,  # 
        xaxis_title='Data',
        yaxis_title='Valor',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98
        ),
        hovermode='x unified'
    )

    fig.show()

def plot_all_results(results, signals):
    """
    Plota os resultados de todos os ativos usando a função plot_backtest_comparison.

    Parâmetros:
        results (dict): Dicionário contendo os resultados de cada ativo (curva de capital).
        signals (dict): Dicionário contendo os DataFrames de sinais de cada ativo.
    """

    for stock, capital_df in results.items():
        if stock in signals:
            signals_df = signals[stock] # Obtém os sinais do ativo
            plot_backtest_comparison(stock, capital_df, signals_df)
        else:
            print(f"Ativo '{stock}' não encontrado em signals.")

def process_stocks(stock_list, all_stock_data):
    """
    Processa uma lista de ativos, realizando previsões, gerando sinais de negociação e executando backtests.
    Retorna um dicionário contendo a curva de capital de cada ativo.

    Parâmetros:
        stock_list (list): Lista de nomes dos ativos.
        all_stock_data (dict): Dicionário contendo os DataFrames de dados de cada ativo.

    Retorna:
        dict: Dicionário contendo a curva de capital de cada ativo.
    """

    results = {}  # Dicionário para armazenar os resultados
    signals = {}  # Dicionário para armazenar os resultados

    # Predições em todos ativos
    for stock in stock_list:
        print(f'Processando {stock}...')
        # Fazer as predições com os ultimos 20% dos dados ( dados de teste)
        df = make_predictions(stock, all_stock_data[stock])
        # Gerar sinais de de negociação
        signals_df = generate_signals(df)
        # Armazenar os sinais no dicionário
        signals[stock] = signals_df
        # Executar o backtest
        _, capital_df = run_backtest(signals_df) # ignora o valor final do capital
        # Armazenar a curva de capital no dicionário
        results[stock] = capital_df
        print(f'{stock} processado!')

    return results, signals

def calculate_final_results(results, signals, initial_money=10000):
    """
    Calcula o resultado final para cada ativo e retorna um DataFrame com os resultados,
    formatando os valores em R$ com duas casas decimais e adicionando a coluna de ganho real.

    Parâmetros:
        results (dict): Dicionário contendo os DataFrames de curva de capital de cada ativo.
        signals (dict): Dicionário contendo os DataFrames de sinais de cada ativo.
        initial_money (float): Quantia inicial em dinheiro para o investimento de comparação. Padrão: 10.000.

    Retorna:
        pd.DataFrame: DataFrame com os resultados finais para cada ativo.
    """

    final_results = []

    for stock, capital_df in results.items():
        if stock in signals:
            signals_df = signals[stock]
            final_capital = capital_df['capital'].iloc[-1]  # Valor final da curva de capital

            # Calculando o valor final do investimento inicial no ativo (buy and hold)
            initial_investment = initial_money * (signals_df['close'].iloc[-1] / signals_df['close'].iloc[0])

            retorno_estrategia = ((final_capital - initial_money) / initial_money) * 100
            retorno_buy_hold = ((initial_investment - initial_money) / initial_money) * 100

            final_results.append({
                'Ativo': stock,
                'Estratégia (Modelo 2) (R$)': f'R$ {final_capital:.2f}',
                'Buy & Hold (R$)': f'R$ {initial_investment:.2f}',
                'Retorno Estratégia (%)': f'{retorno_estrategia:.2f}%',
                'Retorno Buy & Hold (%)': f'{retorno_buy_hold:.2f}%',
            })
        else:
            print(f"Ativo '{stock}' não encontrado em signals.")

    return pd.DataFrame(final_results)