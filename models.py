from utils import *
from config import OUTPUT_DIR  # Importa a configuração da pasta de dados

import joblib 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, save_model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import utils

# Modelo 1 - LSTM Simples
def model_1(stock, df, target_column, learning_rate=0.001, epochs=50, batch_size=32):
    """
    Treina o LSTM e retorna as métricas de desempenho.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados.
        target_column (str): Nome da coluna de destino (target).
        learning_rate (float): Taxa de aprendizado do otimizador Adam. Padrão: 0.001
        epochs (int): Número de épocas para treinamento. Padrão: 50
        batch_size (int): Tamanho do batch para treinamento. Padrão: 32

    Retorna:
        dict: Dicionário com métricas de desempenho (Loss, MSE, RMSE, MAE, MAPE, R²).
    """

    # Separando as variáveis independentes (X) e dependente (y)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # Dividindo os dados em 80% para treino e 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Escalonando os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Salvando o scaler
    joblib.dump(scaler, f"{OUTPUT_DIR}{stock}_scaler_model_1.pkl")

    # Reshape apenas para LSTMs (timesteps=1, features=n_features)
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

    # Configuração de EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Treinamento do modelo
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    # Salvando o modelo treinado
    save_model(model, f"{OUTPUT_DIR}{stock}_model_1.h5")

    # Obtendo a última loss registrada
    loss = history.history['loss'][-1]

    # Previsão no conjunto de teste
    y_pred = model.predict(X_test_scaled).flatten()

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

# Modelo 2 - LSTM com múltiplas camadas e Dropout
def model_2(stock, df, target_column, learning_rate=0.001, dropout_rate=0.03, epochs=50, batch_size=32):
    """
    Treina um modelo LSTM com múltiplas camadas e Dropout e retorna as métricas de desempenho.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados.
        target_column (str): Nome da coluna de destino (target).
        learning_rate (float): Taxa de aprendizado do otimizador Adam. Padrão: 0.001
        dropout_rate (float): Taxa de dropout para regularização. Padrão: 0.03
        epochs (int): Número de épocas para treinamento. Padrão: 50
        batch_size (int): Tamanho do batch para treinamento. Padrão: 32

    Retorna:
        dict: Dicionário com métricas de desempenho (Loss, MSE, RMSE, MAE, MAPE, R²).
    """

    # Separando as variáveis independentes (X) e dependente (y)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # Dividindo os dados em 80% para treino e 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Escalonando os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Salvando o scaler
    joblib.dump(scaler, f"{OUTPUT_DIR}{stock}_scaler_model_2.pkl")

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

    # Configuração de EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Treinamento do modelo
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # Salvando o modelo treinado
    save_model(model, f"{OUTPUT_DIR}{stock}_lstm_model_2.h5")

    # Obtendo a última loss registrada
    loss = history.history['loss'][-1]

    # Previsão no conjunto de teste
    y_pred = model.predict(X_test_scaled).flatten()

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

# Modelo 3
def model_3(stock, df, target_column, learning_rate=0.001, epochs=50, batch_size=32):
    """
    Treina um modelo de Rede Neural Simples (MLP) e retorna as métricas de desempenho.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados.
        target_column (str): Nome da coluna de destino (target).
        learning_rate (float): Taxa de aprendizado do otimizador. Padrão: 0.001
        epochs (int): Número de épocas para treinamento. Padrão: 50
        batch_size (int): Tamanho do batch para treinamento. Padrão: 32

    Retorna:
        dict: Dicionário com métricas de desempenho (Loss, MSE, RMSE, MAE, MAPE, R²).
    """

    # Separando as variáveis independentes (X) e dependente (y)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # Dividindo os dados em 80% para treino e 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Escalonando os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Salvando o scaler
    joblib.dump(scaler, f"{OUTPUT_DIR}{stock}_scaler_model_3.pkl")

    # Criando o modelo MLP
    model = Sequential([
        Input(shape=(X.shape[1],)),   # Ajustando o input
        Dense(32, activation='relu'),  # Primeira camada oculta
        Dense(8, activation='relu'),   # Segunda camada oculta
        Dense(1)                       # Camada de saída
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
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    # Salvando o modelo treinado
    save_model(model, f"{OUTPUT_DIR}{stock}_lstm_model_3.h5")

    # Obtendo a última loss registrada
    loss = history.history['loss'][-1]

    # Previsão no conjunto de teste
    y_pred = model.predict(X_test_scaled).flatten()

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
def model_4(stock, df, target_column):
    """
    Treina um modelo de Regressão Linear e retorna as métricas de desempenho.

    Parâmetros:
        df (pd.DataFrame): DataFrame com os dados.
        target_column (str): Nome da coluna de destino (target).

    Retorna:
        dict: Dicionário com métricas de desempenho (Loss, MSE, RMSE, MAE, MAPE, R²).
    """
    # Verificando se o target_column existe no DataFrame
    if target_column not in df.columns:
        raise ValueError(f"A coluna '{target_column}' não existe no DataFrame.")
    
    # Separando as variáveis independentes (X) e dependente (y)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # Verificando se o DataFrame tem dados suficientes
    if len(X) == 0 or len(y) == 0:
        raise ValueError("O DataFrame está vazio. Não é possível treinar o modelo.")
    
    # Dividindo os dados em 80% para treino e 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Escalonando os dados (MinMaxScaler)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Criando o modelo de regressão linear
    model = LinearRegression()

    # Treinando o modelo
    model.fit(X_train_scaled, y_train)

    # Salvando o modelo treinado
    joblib.dump(model, f"{OUTPUT_DIR}{stock}_model_4.pkl")

    # Salvando o scaler
    joblib.dump(scaler, f"{OUTPUT_DIR}{stock}_scaler_model_4.pkl")

    # Previsão no conjunto de teste
    y_pred = model.predict(X_test)

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


