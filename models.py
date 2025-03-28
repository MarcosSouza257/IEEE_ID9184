from utils import *
from config import *


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Modelo 1 - LSTM Simples
def model_1(stock, df, target_column, num_combination, save_artifacts=False, learning_rate=0.001, epochs=50, batch_size=32):
    """
    Treina um modelo LSTM e, opcionalmente, salva o modelo treinado, scaler_X e scaler_y na pasta do ativo.
    Opcionalmente, salva o modelo treinado e os scalers (scaler_X e scaler_y) na pasta do ativo.

    Parâmetros:
        stock (str): Nome da ação.
        df (pd.DataFrame): DataFrame com os dados.
        target_column (str): Nome da coluna de destino (target).
        num_combination (int): Identificador da combinação de hiperparâmetros.
        save_artifacts (bool): Se True, salva o modelo e os scalers. Padrão: False
        learning_rate (float): Taxa de aprendizado do otimizador. Padrão: 0.001
        epochs (int): Número de épocas para treinamento. Padrão: 50
        batch_size (int): Tamanho do batch para treinamento. Padrão: 32        

    Retorna:
        dict: Dicionário com métricas de desempenho (Loss, MSE, RMSE, MAE, MAPE, R²).
    """
    # Definir semente para reprodutibilidade
    set_seed()

    # Criando diretório específico para o ativo dentro de OUTPUT_DIR
    stock_dir = os.path.join(OUTPUT_DIR, stock)
    os.makedirs(stock_dir, exist_ok=True)

    # Separando variáveis independentes (X) e dependente (y)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values.reshape(-1, 1)  # Reshape para scaler funcionar corretamente

    # Dividindo os dados em 80% para treino e 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Criando e ajustando os scalers
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Reshape para LSTMs (timesteps=1, features=n_features)
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Criando o modelo LSTM
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        LSTM(16, return_sequences=False),
        Dense(1)
    ])

    # Compilando o modelo
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mse', 'mae', 'mape']
    )

    # Configuração de EarlyStopping para evitar overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Treinamento do modelo
    history = model.fit(
        X_train_scaled, y_train_scaled,  # Usando `y_train_scaled`
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_scaled, y_test_scaled),  # Usando `y_test_scaled`
        callbacks=[early_stopping],
        verbose=1
    )

    if save_artifacts:
        # Salvando os scalers na pasta do ativo
        joblib.dump(scaler_X, os.path.join(stock_dir, f"model_1_comb_{num_combination}_scaler_X.pkl"))
        joblib.dump(scaler_y, os.path.join(stock_dir, f"model_1_comb_{num_combination}_scaler_y.pkl"))
        # Salvando o modelo treinado na pasta do ativo
        model_path = os.path.join(stock_dir, f"model_1_comb_{num_combination}.h5")
        save_model(model, model_path, include_optimizer=True)

    # Obtendo a última loss registrada
    loss = history.history['loss'][-1]

    # Previsão no conjunto de teste
    y_pred_scaled = model.predict(X_test_scaled).flatten()

    # Desnormalizando as previsões
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Cálculo de métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + np.finfo(float).eps))) * 100
    r2 = r2_score(y_test, y_pred)

    # Dicionário de métricas
    metrics_dict = {
        "Loss": loss,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R²": r2,
    }

    return metrics_dict

# Modelo 2 - LSTM com múltiplas camadas e Dropout
def model_2(stock, df, target_column, num_combination, save_artifacts=False, learning_rate=0.001, dropout_rate=0.03, epochs=50, batch_size=32):
    """
    Treina um modelo LSTM com múltiplas camadas e Dropout e salva o modelo treinado, scaler_X e scaler_y na pasta do ativo.
    Opcionalmente, salva o modelo treinado e os scalers (scaler_X e scaler_y) na pasta do ativo.

    Parâmetros:
        stock (str): Nome da ação.
        df (pd.DataFrame): DataFrame com os dados.
        target_column (str): Nome da coluna de destino (target).
        num_combination (int): Identificador da combinação de hiperparâmetros.
        save_artifacts (bool): Se True, salva o modelo e os scalers. Padrão: False
        learning_rate (float): Taxa de aprendizado do otimizador. Padrão: 0.001
        epochs (int): Número de épocas para treinamento. Padrão: 50
        batch_size (int): Tamanho do batch para treinamento. Padrão: 32        

    Retorna:
        dict: Dicionário com métricas de desempenho (Loss, MSE, RMSE, MAE, MAPE, R²).
    """
    # Definir semente para reprodutibilidade
    set_seed()

    # Criando diretório específico para o ativo dentro de OUTPUT_DIR
    stock_dir = os.path.join(OUTPUT_DIR, stock)
    os.makedirs(stock_dir, exist_ok=True)

    # Separando variáveis independentes (X) e dependente (y)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values.reshape(-1, 1)  # Reshape para scaler funcionar corretamente

    # Dividindo os dados em 80% para treino e 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Criando e ajustando os scalers
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Reshape para LSTM (timesteps=1, features=n_features)
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Criando o modelo LSTM com Dropout
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        LSTM(200, activation='tanh', recurrent_activation='sigmoid', return_sequences=True),
        LSTM(300, activation='tanh', recurrent_activation='sigmoid', return_sequences=True),
        LSTM(400, activation='tanh', recurrent_activation='sigmoid', return_sequences=True),
        Dropout(dropout_rate),  # Dropout após a penúltima camada LSTM
        LSTM(400, activation='tanh', recurrent_activation='sigmoid', return_sequences=False),
        Dropout(dropout_rate),  # Dropout após a última camada LSTM
        Dense(1)  # Camada de saída
    ])

    # Compilando o modelo
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mse', 'mae', 'mape']
    )

    # Configuração de EarlyStopping para evitar overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Treinamento do modelo
    history = model.fit(
        X_train_scaled, y_train_scaled,  # Usando `y_train_scaled`
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_scaled, y_test_scaled),  # Usando `y_test_scaled`
        callbacks=[early_stopping],
        verbose=1
    )

    if save_artifacts:
        # Salvando os scalers na pasta do ativo
        joblib.dump(scaler_X, os.path.join(stock_dir, f"model_2_comb_{num_combination}_scaler_X.pkl"))
        joblib.dump(scaler_y, os.path.join(stock_dir, f"model_2_comb_{num_combination}_scaler_y.pkl"))
        # Salvando o modelo treinado na pasta do ativo
        model_path = os.path.join(stock_dir, f"model_2_comb_{num_combination}.h5")
        save_model(model, model_path, include_optimizer=True)

    # Obtendo a última loss registrada
    loss = history.history['loss'][-1]

    # Previsão no conjunto de teste
    y_pred_scaled = model.predict(X_test_scaled).flatten()

    # Desnormalizando as previsões
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Cálculo de métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + np.finfo(float).eps))) * 100
    r2 = r2_score(y_test, y_pred)

    # Dicionário de métricas
    metrics_dict = {
        "Loss": loss,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R²": r2,
    }

    return metrics_dict

# Modelo 3
def model_3(stock, df, target_column, num_combination, save_artifacts=False, learning_rate=0.001, epochs=50, batch_size=32):
    """
    Treina um modelo de Rede Neural Simples (MLP) e retorna as métricas de desempenho.
    Opcionalmente, salva o modelo treinado e os scalers (scaler_X e scaler_y) na pasta do ativo.

    Parâmetros:
        stock (str): Nome da ação.
        df (pd.DataFrame): DataFrame com os dados.
        target_column (str): Nome da coluna de destino (target).
        num_combination (int): Identificador da combinação de hiperparâmetros.
        save_artifacts (bool): Se True, salva o modelo e os scalers. Padrão: False
        learning_rate (float): Taxa de aprendizado do otimizador. Padrão: 0.001
        epochs (int): Número de épocas para treinamento. Padrão: 50
        batch_size (int): Tamanho do batch para treinamento. Padrão: 32        

    Retorna:
        dict: Dicionário com métricas de desempenho (Loss, MSE, RMSE, MAE, MAPE, R²).
    """
    # Definir semente para reprodutibilidade
    set_seed()

    # Criando diretório específico para o ativo dentro de OUTPUT_DIR
    stock_dir = os.path.join(OUTPUT_DIR, stock)
    os.makedirs(stock_dir, exist_ok=True)

    # Separando as variáveis independentes (X) e dependente (y)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values.reshape(-1, 1) #Reshape para scaler funcionar corretamente

    # Dividindo os dados em 80% para treino e 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Escalonando os dados
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler(feature_range=(0,1))
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Criando o modelo MLP
    model = Sequential([
        Input(shape=(X.shape[1],)),  # Ajustando o input
        Dense(32, activation='relu'),  # Primeira camada oculta
        Dense(8, activation='relu'),  # Segunda camada oculta
        Dense(1)  # Camada de saída
    ])

    # Compilando o modelo
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mse', 'mae', 'mape']
    )

    # Configuração de EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Treinamento do modelo
    history = model.fit(
        X_train_scaled, y_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_scaled, y_test_scaled),
        callbacks=[early_stopping],
        verbose=1
    )

    if save_artifacts:
        # Salvando os scalers
        joblib.dump(scaler_X, os.path.join(stock_dir, f"model_3_comb_{num_combination}_scaler_X.pkl"))
        joblib.dump(scaler_y, os.path.join(stock_dir, f"model_3_comb_{num_combination}_scaler_y.pkl"))
        # Salvando o modelo treinado
        model_path = os.path.join(stock_dir, f"model_3_comb_{num_combination}.h5")
        save_model(model, model_path, include_optimizer=True)

    # Obtendo a última loss registrada
    loss = history.history['loss'][-1]

    # Previsão no conjunto de teste
    y_pred_scaled = model.predict(X_test_scaled).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # Cálculo de métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)

    # Dicionário de métricas
    metrics_dict = {
        "Loss": loss,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R²": r2,
    }

    return metrics_dict

# Modelo 4
def model_4(stock, df, target_column, num_combination, save_artifacts=False):
    """
    Treina um modelo de Regressão Linear e retorna as métricas de desempenho.
    Opcionalmente, salva o modelo treinado e os scalers (scaler_X e scaler_y) na pasta do ativo.

    Parâmetros:
        stock (str): Nome da ação.
        df (pd.DataFrame): DataFrame com os dados.
        target_column (str): Nome da coluna de destino (target).
        num_combination (int): Identificador da combinação de hiperparâmetros.
        save_artifacts (bool): Se True, salva o modelo e os scalers. Padrão: False

    Retorna:
        dict: Dicionário com métricas de desempenho (MSE, RMSE, MAE, MAPE, R²).
    """

    # Criando diretório específico para o ativo dentro de OUTPUT_DIR
    stock_dir = os.path.join(OUTPUT_DIR, stock)
    os.makedirs(stock_dir, exist_ok=True)

    # Verificando se o target_column existe no DataFrame
    if target_column not in df.columns:
        raise ValueError(f"A coluna '{target_column}' não existe no DataFrame.")

    # Separando as variáveis independentes (X) e dependente (y)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values.reshape(-1, 1) #Reshape para scaler funcionar corretamente

    # Verificando se o DataFrame tem dados suficientes
    if len(X) == 0 or len(y) == 0:
        raise ValueError("O DataFrame está vazio. Não é possível treinar o modelo.")

    # Dividindo os dados em 80% para treino e 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Escalonando os dados (MinMaxScaler)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Criando o modelo de regressão linear
    model = LinearRegression()

    # Treinando o modelo
    model.fit(X_train_scaled, y_train_scaled)

    if save_artifacts:
        # Salvando o modelo treinado
        model_path = os.path.join(stock_dir, f"model_4_comb_{num_combination}.pkl")
        joblib.dump(model, model_path)
        # Salvando os scalers
        joblib.dump(scaler_X, os.path.join(stock_dir, f"model_4_comb_{num_combination}_scaler_X.pkl"))
        joblib.dump(scaler_y, os.path.join(stock_dir, f"model_4_comb_{num_combination}_scaler_y.pkl"))

    # Previsão no conjunto de teste
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

    # Cálculo de métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    # Cálculo do MAPE, com verificação para evitar divisão por zero
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + np.finfo(float).eps))) * 100

    r2 = r2_score(y_test, y_pred)

    # Dicionário de métricas
    metrics_dict = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R²": r2,
    }

    return metrics_dict


