import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.dates as mdates


def train_lstm(X_train, y_train, input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_split=0.2, verbose=0)
    return model


def mostrar_pagina_lstm():
    # Título de la aplicación
    st.title("Modelo LSTM para Predicción de Precios de Acciones")

    # Descripción de la aplicación
    st.write("""
    Esta aplicación permite realizar predicciones del precio de cierre de acciones utilizando un modelo basado en LSTM.
    """)

    # Entrada del usuario: ticker de la acción
    ticker = st.text_input("Ingrese el ticker de la acción", "FSM")

    # Selección de rango de fechas
    start_date = st.date_input("Fecha de inicio", pd.to_datetime("2019-01-01"))
    end_date = st.date_input("Fecha de fin", pd.to_datetime("2023-12-31"))

    # Descargar los datos de Yahoo Finance
    st.write("### Descargando datos...")
    data = yf.download(ticker, start=start_date, end=end_date)

    # Mostrar los datos descargados
    st.write("### Datos descargados:")
    st.write(data.head())

    # Graficar los precios reales
    st.write("### Gráfico de los precios reales")
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label='Precio de cierre', color='blue')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio')
    ax.set_title(f'Precio de Cierre de {ticker} a lo largo del tiempo')
    ax.legend()
    st.pyplot(fig)

    # Preparar los datos para el modelo LSTM
    data_close = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_close)

    X, y = [], []
    window_size = 60
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i-window_size:i])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    # Entrenar el modelo LSTM
    st.write("### Entrenando el modelo LSTM...")
    model = train_lstm(X_train, y_train, (X_train.shape[1], X_train.shape[2]))

    # Hacer predicciones
    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Graficar las predicciones vs los valores reales
    st.write("### Comparación entre Predicciones y Valores Reales")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index[-len(predictions):], predictions,
            label='Predicciones', color='green')
    ax.plot(data.index[-len(y_test):], y_test,
            label='Valores Reales', color='blue')
    ax.set_title('Predicción vs Real del Precio de Cierre')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio de Cierre')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Evaluación del modelo
    mape_lstm = mean_absolute_percentage_error(y_test, predictions)
    rmse_lstm = np.sqrt(mean_squared_error(y_test, predictions))

    st.write(f"**MAPE del modelo LSTM:** {mape_lstm:.4f}")
    st.write(f"**RMSE del modelo LSTM:** {rmse_lstm:.4f}")


# Llamar a la función de Streamlit para mostrar la página
if __name__ == "__main__":
    mostrar_pagina_lstm()
