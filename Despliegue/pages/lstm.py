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


def mostrar_pagina_lstm():
    # Título de la aplicación
    st.title("Modelo LSTM para Predicción de Precios de Acciones")

    # Descripción de la aplicación
    st.write("""
    Esta aplicación permite realizar un análisis de predicción de precios de acciones utilizando un modelo de LSTM.
    Puedes seleccionar el ticker de la acción y el rango de fechas, y ver cómo el modelo predice los precios futuros en comparación con los reales.
    """)

    # Entrada del usuario: ticker de la acción
    ticker = st.text_input("Ingrese el ticker de la acción", "FSM")

    # Explicación sobre el ticker
    st.write("""
    **Ticker:** Un ticker es un símbolo único utilizado para identificar una acción en el mercado. Por ejemplo, 'FSM' corresponde a Fortuna Silver Mines Inc.
    """)

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
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['Close']].values)

    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i, 0])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    # Crear el modelo LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,
              input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entrenar el modelo
    st.write("### Entrenando el modelo LSTM...")
    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_split=0.2, verbose=0)

    # Hacer predicciones
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
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

    # Sección para hacer predicciones con datos ingresados
    st.write("### Realizar una Predicción con Nuevos Datos")
    user_input = st.text_input("Ingrese los últimos 60 precios de cierre separados por coma", ",".join(
        map(str, data['Close'][-60:].values)))

    if user_input:
        # Convertir el input a un arreglo
        input_data = np.array([float(x)
                              for x in user_input.split(',')]).reshape(-1, 1)
        input_scaled = scaler.transform(input_data)
        input_scaled = np.reshape(input_scaled, (1, input_scaled.shape[0], 1))

        # Realizar la predicción
        prediction_scaled = model.predict(input_scaled)
        prediction = scaler.inverse_transform(prediction_scaled)

        st.write(
            f"La predicción para el precio de cierre del siguiente día es: {prediction[0][0]}")


# Llamar a la función de Streamlit para mostrar la página
if __name__ == "__main__":
    mostrar_pagina_lstm()
